#ifndef VJPHPP
#define VJPHPP
/* ------------------------------------------------------------
   Synthesis for a gradient-based caller (e.g. torch autograd):

   - SynManyStokesIO: Stokes only, with the arithmetic of ME_Synth_RF
     (bit-identical Stokes), written straight into the caller's
     buffers in any layout and in the caller's dtype.
   - SynManyVJPIO: vector-Jacobian product g_p = sum_{s,w} RF[p][s][w] g[s][w]
     per pixel, without materialising the (npix, 9, 4, nWav) RF array.
     mode 1 (default): reverse mode through the Stokes formulae, the
     absorption matrix and the Zeeman profiles; mode 0, and always with a
     spectral PSF: the pixel's RF in a thread workspace, then a dot product.
   ------------------------------------------------------------ */
#include <vector>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "Milne.hpp"

namespace ml{

  // ---------------------------------------------------------------- //
  // Absorption matrix with the arithmetic of compute_K_der (so the
  // Stokes match ME_Synth_RF bit for bit), without any derivative.
  // buff: >= 15*nWav, zeroed by the caller.

  template<typename T>
  void compute_K_rfarith(int const nWav, const double* __restrict__ wav, T* const __restrict__ buff,
			 T const B, T const vl, T const vD, T const eta, T const dam,
			 std::vector<ln::line<T>> const& lines, T const sinin, T const cosin,
			 T const sin2az, T const cos2az)
  {
    T const sinin2   = sinin*sinin;
    T const cosin2_1 = ((cosin*cosin) + 1)/2;

    T* const __restrict__ Ki = &buff[0*nWav];
    T* const __restrict__ Kq = &buff[1*nWav];
    T* const __restrict__ Ku = &buff[2*nWav];
    T* const __restrict__ Kv = &buff[3*nWav];
    T* const __restrict__ Fq = &buff[4*nWav];
    T* const __restrict__ Fu = &buff[5*nWav];
    T* const __restrict__ Fv = &buff[6*nWav];
    T* const __restrict__ H  = &buff[7*nWav];
    T* const __restrict__ F  = &buff[10*nWav];
    T* const __restrict__ h  = &buff[13*nWav];
    T* const __restrict__ f  = &buff[14*nWav];

    for(int ww = 0; ww<nWav; ++ww) Ki[ww] = 1;

    int const nLines = int(lines.size());
    for(int ii=0; ii<nLines; ++ii){
      ln::line<T> const& iLine = lines[ii];
      std::memset(&buff[7*nWav], 0, 8*nWav*sizeof(T));

      int const nComp  = int(iLine.splitting.size());
      T const wav0 = iLine.w0;
      T const va   = iLine.w0 * (vl / (phyc::CC*1.e-5));
      int const l0 = iLine.l0, l1 = iLine.l1, nWav_line = l1-l0+1;

      for(int kk=0; kk<nComp; ++kk){
	int const iL = iLine.iL[kk];
	T const vb   = -iLine.splitting[kk] * B;
	T const str  =  iLine.strength[kk];
	pr::compute_profile<T>(nWav_line, &wav[l0], wav0, str, va, dam, vD, vb, &h[l0], &f[l0]);
	int const off = iL*nWav;
	for(int ww=l0; ww <= l1; ++ww){
	  H[off + ww] +=     h[ww];
	  F[off + ww] += 2 * f[ww];
	}
      }

      T const eta2 = iLine.gf * eta;
      for(int ww=l0; ww <= l1; ++ww){
	int const w0 = ww, w1 = nWav + ww, w2 = 2*nWav + ww;
	T const hsum = H[w0] + H[w2];
	T const fsum = F[w0] + F[w2];
	T const hsub = H[w2] - H[w0];
	T const fsub = F[w2] - F[w0];
	T tmp0 = (H[w1] * sinin2 + hsum * cosin2_1)/2;
	T tmp  = (H[w1] - hsum/2)/2;
	T tmp1 = (F[w1] - fsum/2)/2;
	T tmp2 = hsub/2;
	T tmp3 = fsub/2;
	tmp0 *= eta2, tmp *= eta2, tmp1 *= eta2, tmp2 *= eta2, tmp3 *= eta2;
	Ki[ww] += tmp0;
	Kq[ww] += tmp  * cos2az * sinin2;
	Fq[ww] += tmp1 * cos2az * sinin2;
	Ku[ww] += tmp  * sin2az * sinin2;
	Fu[ww] += tmp1 * sin2az * sinin2;
	Kv[ww] += tmp2 * cosin;
	Fv[ww] += tmp3 * cosin;
      }
    }
  }

  // ---------------------------------------------------------------- //
  // Stokes only, same numbers as ME_Synth_RF's Stokes. ws >= 15*nWav.

  template<typename T>
  void ME_Synth_rfarith(std::vector<double> const &wav, std::vector<ln::line<T>> const& lines,
			const T* const __restrict__ m, T* const __restrict__ out, T const mu,
			T* const __restrict__ ws)
  {
    int const nWav = int(wav.size());
    std::memset(ws, 0, 15*nWav*sizeof(T));
    T const sinin = sin(m[1]), cosin = cos(m[1]), sin2az = sin(2*m[2]), cos2az = cos(2*m[2]);
    compute_K_rfarith<T>(nWav, &wav[0], ws, m[0], m[3], m[4], m[5], m[6], lines, sinin, cosin, sin2az, cos2az);
    compute_Stokes<T>(nWav, ws, out, m[7], m[8], mu);
  }

  // ---------------------------------------------------------------- //
  // Full reverse-mode VJP. Forward sweep: Zeeman profiles (kept), H/F per
  // line, K. Then the Stokes adjoint gives K̄; the line adjoint gives
  // the eta/inc/azi gradients and H̄, F̄; each component's profile
  // derivatives are contracted with H̄, F̄ on the fly. No dK array.

  template<typename T>
  inline size_t ws_size_rev(std::vector<ln::line<T>> const& lines, int const nWav)
  {
    size_t nc = 0;
    for(auto const& l: lines) nc += l.splitting.size();
    return size_t(nWav)*(14 + 6 + 6*lines.size() + 2*nc);
  }

  template<typename T>
  void ME_VJP_rev(std::vector<double> const &wav, std::vector<ln::line<T>> const& lines,
		  const T* const __restrict__ m, const T* const __restrict__ g, T* const __restrict__ gout,
		  T const mu, T* const __restrict__ ws)
  {
    int const nWav = int(wav.size());
    int const nLines = int(lines.size());
    T const B = m[0], inc = m[1], azi = m[2], vl = m[3], vD = m[4], eta = m[5], dam = m[6], S1 = m[8];
    T const sinin = sin(inc), cosin = cos(inc), sin2az = sin(2*azi), cos2az = cos(2*azi);
    T const sinin2 = sinin*sinin, cosin2_1 = ((cosin*cosin) + 1)/2;
    T const sin2in = sin(2*inc), cosdi = -sinin, sinda = 2*cos2az, cosda = -2*sin2az;

    T* const __restrict__ K  = ws;                 // 7 nWav
    T* const __restrict__ Kb = ws + 7*nWav;        // 7 nWav
    T* const __restrict__ HB = ws + 14*nWav;       // 3 nWav  (H̄), then F̄ 3 nWav
    T* const __restrict__ FB = ws + 17*nWav;
    T* const __restrict__ HFall = ws + 20*nWav;    // per line: H 3 nWav, F 3 nWav
    T* const __restrict__ hfall = HFall + 6*size_t(nLines)*nWav;   // per component: h, f

    std::memset(K, 0, 7*nWav*sizeof(T));
    for(int ww=0; ww<nWav; ++ww) K[ww] = 1;

    // --- forward sweep --- //
    {
      T* hf = hfall;
      for(int ii=0; ii<nLines; ++ii){
	ln::line<T> const& iLine = lines[ii];
	T* const __restrict__ H = HFall + 6*size_t(ii)*nWav;
	T* const __restrict__ F = H + 3*nWav;
	std::memset(H, 0, 6*nWav*sizeof(T));
	int const nComp = int(iLine.splitting.size());
	T const wav0 = iLine.w0, va = iLine.w0 * (vl / (phyc::CC*1.e-5));
	int const l0 = iLine.l0, l1 = iLine.l1, nWl = l1-l0+1;
	for(int kk=0; kk<nComp; ++kk, hf += 2*nWav){
	  int const iL = iLine.iL[kk];
	  T const vb = -iLine.splitting[kk] * B;
	  pr::compute_profile<T>(nWl, &wav[l0], wav0, iLine.strength[kk], va, dam, vD, vb, &hf[l0], &hf[nWav+l0]);
	  T* const __restrict__ Hc = H + iL*nWav;
	  T* const __restrict__ Fc = F + iL*nWav;
	  const T* const __restrict__ h = hf;
	  const T* const __restrict__ f = hf + nWav;
	  for(int ww=l0; ww<=l1; ++ww){ Hc[ww] += h[ww]; Fc[ww] += 2*f[ww]; }
	}
	T const eta2 = iLine.gf * eta;
#pragma omp simd
	for(int ww=l0; ww<=l1; ++ww){
	  T const hsum = H[ww] + H[2*nWav+ww], fsum = F[ww] + F[2*nWav+ww];
	  T const hsub = H[2*nWav+ww] - H[ww], fsub = F[2*nWav+ww] - F[ww];
	  T const tmp0 = eta2*((H[nWav+ww]*sinin2 + hsum*cosin2_1)/2);
	  T const tmp  = eta2*((H[nWav+ww] - hsum/2)/2);
	  T const tmp1 = eta2*((F[nWav+ww] - fsum/2)/2);
	  T const tmp2 = eta2*(hsub/2), tmp3 = eta2*(fsub/2);
	  K[ww]          += tmp0;
	  K[1*nWav+ww]   += tmp  * cos2az * sinin2;
	  K[2*nWav+ww]   += tmp  * sin2az * sinin2;
	  K[3*nWav+ww]   += tmp2 * cosin;
	  K[4*nWav+ww]   += tmp1 * cos2az * sinin2;
	  K[5*nWav+ww]   += tmp1 * sin2az * sinin2;
	  K[6*nWav+ww]   += tmp3 * cosin;
	}
      }
    }

    // --- Stokes adjoint: K̄ --- //
    T a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0, a8=0;
#pragma omp simd reduction(+:a7,a8)
    for(int ww=0; ww<nWav; ++ww){
      T const ki = K[ww], kq = K[nWav+ww], ku = K[2*nWav+ww], kv = K[3*nWav+ww];
      T const fq = K[4*nWav+ww], fu = K[5*nWav+ww], fv = K[6*nWav+ww];
      T const ki2 = ki*ki;
      T const GP3 = ki2 + fq*fq + fu*fu + fv*fv;
      T const GP1 = GP3 - (kq*kq + ku*ku + kv*kv);
      T const GP2 = kq*fq + ku*fu + kv*fv;
      T const del = ki2*GP1 - GP2*GP2;
      T const GP4 = ki2*kq + ki*(kv*fu - ku*fv);
      T const GP5 = ki2*ku + ki*(kq*fv - kv*fq);
      T const GP6 = ki2*kv + ki*(ku*fq - kq*fu);
      T const gi = g[ww], gq = g[nWav+ww], gu = g[2*nWav+ww], gv = g[3*nWav+ww];
      T const Z  = gi*ki*GP3 - gq*(GP4 + fq*GP2) - gu*(GP5 + fu*GP2) - gv*(GP6 + fv*GP2);
      T const idel = T(1)/del;
      T const c  = S1*mu*idel;
      a7 += gi;
      a8 += mu*idel*Z;
      T const bdel = -c*Z*idel;
      T const cI = c*gi, bP4 = -c*gq, bP5 = -c*gu, bP6 = -c*gv;
      T bki = cI*GP3, bGP3 = cI*ki;
      T bfq = bP4*GP2, bfu = bP5*GP2, bfv = bP6*GP2;
      T const bGP2 = bP4*fq + bP5*fu + bP6*fv - 2*GP2*bdel;
      T const bGP1 = bdel*ki2;
      bki += bdel*2*ki*GP1;
      T bkq = bP4*ki2, bku = bP5*ki2, bkv = bP6*ki2;
      bki += bP4*(2*ki*kq + kv*fu - ku*fv) + bP5*(2*ki*ku + kq*fv - kv*fq) + bP6*(2*ki*kv + ku*fq - kq*fu);
      bkv += bP4*ki*fu;  bfu += bP4*ki*kv;  bku -= bP4*ki*fv;  bfv -= bP4*ki*ku;
      bkq += bP5*ki*fv;  bfv += bP5*ki*kq;  bkv -= bP5*ki*fq;  bfq -= bP5*ki*kv;
      bku += bP6*ki*fq;  bfq += bP6*ki*ku;  bkq -= bP6*ki*fu;  bfu -= bP6*ki*kq;
      bkq += bGP2*fq; bfq += bGP2*kq; bku += bGP2*fu; bfu += bGP2*ku; bkv += bGP2*fv; bfv += bGP2*kv;
      bGP3 += bGP1;
      bkq -= 2*kq*bGP1; bku -= 2*ku*bGP1; bkv -= 2*kv*bGP1;
      bki += 2*ki*bGP3; bfq += 2*fq*bGP3; bfu += 2*fu*bGP3; bfv += 2*fv*bGP3;
      Kb[ww] = bki; Kb[nWav+ww] = bkq; Kb[2*nWav+ww] = bku; Kb[3*nWav+ww] = bkv;
      Kb[4*nWav+ww] = bfq; Kb[5*nWav+ww] = bfu; Kb[6*nWav+ww] = bfv;
    }

    // --- line and component adjoints --- //
    {
      const T* hf = hfall;
      for(int ii=0; ii<nLines; ++ii){
	ln::line<T> const& iLine = lines[ii];
	const T* const __restrict__ H = HFall + 6*size_t(ii)*nWav;
	const T* const __restrict__ F = H + 3*nWav;
	int const l0 = iLine.l0, l1 = iLine.l1;
	T const rat = iLine.gf, eta2 = rat*eta, lo = eta2/2;
#pragma omp simd reduction(+:a1,a2,a5)
	for(int ww=l0; ww<=l1; ++ww){
	  T const kbi = Kb[ww], kbq = Kb[nWav+ww], kbu = Kb[2*nWav+ww], kbv = Kb[3*nWav+ww];
	  T const kbfq = Kb[4*nWav+ww], kbfu = Kb[5*nWav+ww], kbfv = Kb[6*nWav+ww];
	  T const hsum = H[ww] + H[2*nWav+ww], fsum = F[ww] + F[2*nWav+ww];
	  T const hsub = H[2*nWav+ww] - H[ww], fsub = F[2*nWav+ww] - F[ww];
	  T const t0 = (H[nWav+ww]*sinin2 + hsum*cosin2_1)/2;
	  T const t  = (H[nWav+ww] - hsum/2)/2;
	  T const t1 = (F[nWav+ww] - fsum/2)/2;
	  T const t2 = hsub/2, t3 = fsub/2;
	  T const Aq = cos2az*kbq + sin2az*kbu, Af = cos2az*kbfq + sin2az*kbfu;
	  a5 += rat*(t0*kbi + t*sinin2*Aq + t2*cosin*kbv + t1*sinin2*Af + t3*cosin*kbfv);
	  a1 += eta2*(t*sin2in*(kbi + Aq) + t1*sin2in*Af + (t2*kbv + t3*kbfv)*cosdi);
	  a2 += eta2*sinin2*(t*(cosda*kbq + sinda*kbu) + t1*(cosda*kbfq + sinda*kbfu));
	  T const A = sinin2*Aq, Ab = sinin2*Af;
	  HB[nWav+ww]   = lo*(sinin2*kbi + A);
	  HB[ww]        = lo*(cosin2_1*kbi - A/2 - cosin*kbv);
	  HB[2*nWav+ww] = lo*(cosin2_1*kbi - A/2 + cosin*kbv);
	  FB[nWav+ww]   = lo*Ab;
	  FB[ww]        = lo*(-Ab/2 - cosin*kbfv);
	  FB[2*nWav+ww] = lo*(-Ab/2 + cosin*kbfv);
	}
	int const nComp = int(iLine.splitting.size());
	T const wav0 = iLine.w0, va = iLine.w0 * (vl / (phyc::CC*1.e-5));
	T const dvadvlos = -iLine.w0 / (phyc::CC*1.e-5) / vD;
	for(int kk=0; kk<nComp; ++kk, hf += 2*nWav){
	  int const iL = iLine.iL[kk];
	  T const vb = -iLine.splitting[kk] * B, dvB = -iLine.splitting[kk] / vD;
	  T const strnpi = iLine.strength[kk] / phyc::SQPI;
	  const T* const __restrict__ h = hf;
	  const T* const __restrict__ f = hf + nWav;
	  const T* const __restrict__ hb = HB + iL*nWav;
	  const T* const __restrict__ fb = FB + iL*nWav;
	  T u0 = 0, u1 = 0, u2 = 0;
#pragma omp simd reduction(+:u0,u1,u2)
	  for(int ww=l0; ww<=l1; ++ww){
	    T const Vld  = (wav[ww] - wav0 - va + vb) / vD;
	    T const dhdv = 4*dam*f[ww] - 2*Vld*h[ww];
	    T const dfdv = 2*(strnpi - dam*h[ww] - 2*Vld*f[ww]);
	    T const u = hb[ww]*dhdv + fb[ww]*dfdv;
	    u0 += u;
	    u1 += u*(-Vld/vD);
	    u2 += fb[ww]*dhdv - hb[ww]*dfdv;
	  }
	  a0 += u0*dvB; a3 += u0*dvadvlos; a4 += u1; a6 += u2;
	}
      }
    }
    gout[0]=a0; gout[1]=a1; gout[2]=a2; gout[3]=a3; gout[4]=a4; gout[5]=a5; gout[6]=a6; gout[7]=a7; gout[8]=a8;
  }

  template<typename T>
  inline bool has_degradation(Milne<T> const& me)
  {
    for(auto const& it: me.get_regions()) if(it.Degradation.npad != 0) return true;
    return false;
  }
}

namespace wr{

  // ---------------------------------------------------------------- //
  // Strided, mixed-precision I/O: T is the synthesis precision, U the
  // caller's tensor dtype. Stokes s of pixel p, wavelength w lives at
  // out[s][p*ps + w*ls]; the same for the upstream gradient.

  template<typename T, typename U>
  void SynManyStokesIO(std::vector<ml::Milne<T>> const& ME, const T* __restrict__ m_in,
		       U* const out[4], long const ps, long const ls, long const ny, long const nx, T const mu)
  {
    long const nthreads = int(ME.size()), nPix = nx*ny;
    int  const nWav = ME[0].get_number_of_wavelength();
#pragma omp parallel default(shared) num_threads(nthreads)
    {
      int const tid = omp_get_thread_num();
      std::vector<T> ws(15*nWav), st(4*nWav);
      T mm[9];
#pragma omp for
      for(long ipix = 0; ipix<nPix; ++ipix){
	std::memcpy(mm, &m_in[ipix*9], 9*sizeof(T));
	ME[tid].checkParameters(mm);
	ml::ME_Synth_rfarith<T>(ME[tid].get_wavelength(), ME[tid].get_lines(), mm, st.data(), mu, ws.data());
	for(auto const& it: ME[tid].get_regions())
	  for(int ss=0; ss<4; ++ss) it.Degrade(&st[ss*nWav]);
	for(int ss=0; ss<4; ++ss){
	  U* const o = out[ss] + ipix*ps;
	  for(int ww=0; ww<nWav; ++ww) o[ww*ls] = U(st[ss*nWav+ww]);
	}
      }
    }
  }

  template<typename T, typename U>
  void SynManyVJPIO(std::vector<ml::Milne<T>> const& ME, const T* __restrict__ m_in,
		    const U* const grad[4], long const ps, long const ls, T* __restrict__ gout,
		    long const ny, long const nx, T const mu, int const mode)
  {
    long const nthreads = int(ME.size()), nPix = nx*ny;
    int  const nWav = ME[0].get_number_of_wavelength();
    long const sStride = 4*nWav;
    bool const fused = (mode == 1) && !ml::has_degradation(ME[0]);   // a spectral PSF needs the per-pixel RF
#pragma omp parallel default(shared) num_threads(nthreads)
    {
      int const tid = omp_get_thread_num();
      std::vector<T> ws(ml::ws_size_rev(ME[0].get_lines(), nWav)), st(4*nWav), rf(36*nWav), g(4*nWav);
      T mm[9];
#pragma omp for
      for(long ipix = 0; ipix<nPix; ++ipix){
	std::memcpy(mm, &m_in[ipix*9], 9*sizeof(T));
	ME[tid].checkParameters(mm);
	for(int ss=0; ss<4; ++ss){
	  if(grad[ss] == nullptr){ for(int ww=0; ww<nWav; ++ww) g[ss*nWav+ww] = 0; continue; }
	  const U* const gi = grad[ss] + ipix*ps;
	  for(int ww=0; ww<nWav; ++ww) g[ss*nWav+ww] = T(gi[ww*ls]);
	}
	T* const go = &gout[9*ipix];
	if(fused){
	  ml::ME_VJP_rev<T>(ME[tid].get_wavelength(), ME[tid].get_lines(), mm, g.data(), go, mu, ws.data());
	}else{
	  ME[tid].synthesize_rf(mm, st.data(), rf.data(), mu);
	  for(int pp=0; pp<9; ++pp){
	    const T* const r = &rf[pp*sStride];
	    T acc = 0;
#pragma omp simd reduction(+:acc)
	    for(long jj=0; jj<sStride; ++jj) acc += r[jj]*g[jj];
	    go[pp] = acc;
	  }
	}
      }
    }
  }
}

#endif
