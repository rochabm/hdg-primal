********************************************************************************
*              SUBROUTINES FOR TEN TUSSCHER 2006 CELL MODEL                    *
********************************************************************************     
            
c ------------------------------------------------------------------------------      
      subroutine tt2006_init(neq, sv)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 sv
      dimension sv(19)
c
      sv( 1) = -85.23d0
      sv( 2) = 0.00621d0
      sv( 3) = 0.4712d0
      sv( 4) = 0.0095d0   
      sv( 5) = 0.00172d0
      sv( 6) = 0.7444d0
      sv( 7) = 0.7045d0 
      sv( 8) = 3.373e-5
      sv( 9) = 0.7888d0   
      sv(10) = 0.9755d0   
      sv(11) = 0.9953d0  
      sv(12) = 0.999998d0 
      sv(13) = 2.42e-8 
      sv(14) = 0.000126d0
      sv(15) = 3.64d0
      sv(16) = 0.00036d0
      sv(17) = 0.9073d0
      sv(18) = 8.604d0 
      sv(19) = 136.89d0
c
      return
      end
      
c ------------------------------------------------------------------------------      
      subroutine tt2006_equation(neq, dt, sv, dsv, istim)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 dt, istim
c      
c     variaveis
c      
      real*8 V,Xr1,Xr2,Xs,m,h,j,d,f,f2,fCass,s,r
      real*8 Cai,CaSR,Cass,Rprime,Nai,Ki
c
c     derivadas
c
      real*8 d_dt_V,d_dt_Xr1,d_dt_Xr2,d_dt_Xs,d_dt_m,d_dt_h,d_dt_j
      real*8 d_dt_d,d_dt_f,d_dt_f2,d_dt_fCass,d_dt_s,d_dt_r,d_dt_Cai
      real*8 d_dt_CaSR,d_dt_Cass,d_dt_Rprime,d_dt_Nai,d_dt_Ki
c
c     constantes
c
      real*8 RR,T,FF,Cm,V_c,Ko,Nao,Cao,P_kna,K_mk,P_NaK,K_mNa,K_pCa
      real*8 V_rel,k1_prime,max_sr,min_sr,EC,Vmax_up
      real*8 alpha,gamma,K_sat,Km_Ca,Km_Nai,K_NaCa
      real*8 g_to,g_Kr,g_Ks,g_CaL,g_Na,g_pK,g_bca,g_pCa,g_K1,g_bna
      real*8 xIstim,k3
      real*8 var_Ca_K_up, var_Ca_V_xfer, var_Ca_k4, var_Ca_Buf_c
      real*8 var_Ca_K_buf_c, var_Ca_k2_prime, var_Ca_K_buf_sr
      real*8 var_Ca_Buf_sr, var_Ca_Buf_ss, var_Ca_K_buf_ss
      real*8 var_Ca_V_sr, var_Ca_V_ss
c
c     valores calculados
c      
      real*8 EK, EKs, ENa, ECa, beta_k1, alpha_K1, xK1_inf
      real*8 IK1,Ito,IKr,IKs,IpK,ICaL,IbCa,IpCa,INa,INaK,IbNa,INaCa
      real*8 xr1_inf, alpha_xr1, beta_xr1, tau_xr1
      real*8 xr2_inf, alpha_xr2, beta_xr2, tau_xr2
      real*8 m_inf,alpha_m,beta_m,tau_m
      real*8 xs_inf,alpha_xs,beta_xs,tau_xs
      real*8 h_inf,alpha_h,beta_h,tau_h
      real*8 j_inf,alpha_j,beta_j,tau_j
      real*8 d_inf,alpha_d,beta_d,gamma_d,tau_d
      real*8 f_inf,tau_f,f2_inf,tau_f2
      real*8 fCass_inf,tau_fCass,s_inf,tau_s
      real*8 r_inf,tau_r
      real*8 kcasr,k1,var_calcium_dynamics__o,Irel
c
c     valores calculados - dinamicas
c 
      real*8 var_Ca_V_leak, var_Ca_i_up, var_Ca_i_leak
      real*8 var_Ca_i_xfer, var_Ca_k2, Ca_i_bufc, Ca_ss_bufss
      real*8 var_Ca_Ca_sr_bufsr, var_Ca_V_c, var_Ca_F
      real*8 var_Ca_Cm, var_Ca_ICaL, var_Ca_INaCa
      real*8 var_Ca_IbCa, var_Ca_IpCa
c
c     vetores
c      
      real*8 sv,dsv      
c      
      dimension sv(19), dsv(19)
c
      V = sv(1)
      Xr1 = sv(2)
      Xr2 = sv(3)   
      Xs = sv(4)    
      m = sv(5)      
      h = sv(6)     
      j = sv(7)     
      d = sv(8)      
      f = sv(9)     
      f2 = sv(10)    
      fCass = sv(11)
      s = sv(12)  
      r = sv(13)     
      Cai = sv(14)
      CaSR = sv(15)
      Cass = sv(16)
      Rprime = sv(17)
      Nai = sv(18)
      Ki = sv(19)
c     
c     constantes
c     
      RR  = 8314.472d0
      T   = 310.0d0
      FF  = 96485.3415d0
      Cm  = 0.185d0
      V_c = 0.016404d0

      Ko  = 5.4d0
      Nao = 140.0d0
      Cao = 2.0d0
      
      P_kna = 0.03d0
      K_mk  = 1.0d0
      P_NaK = 2.724d0
      K_mNa = 40.0d0
      K_pCa = 0.0005d0
c      
c     Calcium dynamics
c      
      V_rel    = 0.102d0
      k1_prime = 0.15d0
      max_sr   = 2.5d0
      min_sr   = 1.0d0
      EC       = 1.5d0
      Vmax_up  = 0.006375d0
c      
c     NCX consts
c      
      alpha  = 2.5d0
      gamma  = 0.35d0
      K_sat  = 0.1d0
      Km_Ca  = 1.38d0
      Km_Nai = 87.5d0
      K_NaCa = 1000.0d0
      
      g_to  = 0.294d0
      g_Kr  = 0.153d0
      g_Ks  = 0.098d0
      g_CaL = 3.98e-05
      g_Na  = 14.838d0
      g_pK  = 0.0146d0
      g_bca = 0.000592d0
      g_pCa = 0.1238d0
      g_K1  = 5.405d0
      g_bna = 0.00029d0
c
c     calculations
c     
      EK  = ((RR * T) / FF) * dlog(Ko / Ki)
      EKs = ((RR * T) / FF) * dlog((Ko+(P_kna*Nao))/(Ki+(P_kna*Nai)))
      ENa = ((RR * T) / FF) * dlog(Nao / Nai)
      ECa = ((0.5d0 * RR * T) / FF) * dlog(Cao / Cai)
c      
      beta_K1 = ((3.0d0 * dexp(0.0002d0 * ((V - EK) + 100.0d0))) +
     &      dexp(0.1d0 * ((V-EK)-10.0d0)))/(1.0d0+dexp((-0.5d0)*(V-EK)))      
      alpha_K1 = 0.1d0 / (1.0d0 + dexp(0.06d0 * ((V - EK) - 200.0d0)))
      xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
c      
      IK1 = g_K1 * xK1_inf * (V - EK)
      Ito = g_to * r * s * (V - EK)
      IKr = g_Kr * Xr1 * Xr2 * (V - EK) * dsqrt(Ko / 5.4d0)
      IKs = g_Ks * (Xs**2) * (V - EKs)
      IpK = (g_pK * (V - EK)) / (1.0d0 + dexp((25.0d0 - V) / 5.98d0))
c      
      if ( (V<15.0-1.0e-5).or.(V>15.0+1.0e-5) ) then 
         ICaL = ((((g_CaL * d * f * f2 * fCass * 4.0 * (V - 15.0) *
     &        (FF**2)) / (RR * T)) * ((0.25 * Cass *
     &        dexp((2.0 * (V - 15.0) * FF) / (RR * T))) - Cao)) /
     &        (exp((2.0 * (V - 15.0) * FF) / (RR * T)) - 1.0))
      else
         ICaL = g_CaL*d*f*f2*fCass*2.0d0*FF*(0.25d0*Cass-Cao)
      end if
      
      IbCa = g_bca * (V - ECa)
      IpCa = (g_pCa * Cai) / (Cai + K_pCa)
c      
      INaK = ((((P_NaK*Ko) / (Ko+K_mk))*Nai) / (Nai+K_mNa)) /
     &     (1.0d0 + (0.1245d0 * dexp(((-0.1d0)*V*FF) / (RR*T))) +
     &     (0.0353d0 * dexp(((-V)*FF)/(RR*T))))      
      INa  = g_Na * (m**3) * h * j * (V - ENa)
      IbNa = g_bna * (V - ENa)
      INaCa = (K_NaCa * ((dexp((gamma*V*FF)/(RR*T))*(Nai**3)*Cao)
     &     - (dexp(((gamma-1.0d0)*V*FF)/(RR*T))*(Nao**3)*Cai*alpha))) /
     &       (( (Km_Nai**3) + (Nao**3))*(Km_Ca+Cao) *
     &       (1.0d0 + (K_sat*dexp(((gamma-1.0d0)*V*FF)/(RR*T)))))
c
c     stimulus
c      
      xIstim = istim
c
c     compute HH-currents
c
      xr1_inf   = 1.0d0 / (1.0d0 + dexp(((-26.0d0) - V) / 7.0d0))
      alpha_xr1 = 450.0d0 / (1.0d0 + dexp(((-45.0d0) - V) / 10.0d0))
      beta_xr1  = 6.0d0 / (1.0d0 + dexp((V + 30.0d0) / 11.5d0))
      tau_xr1   = 1.0d0 * alpha_xr1 * beta_xr1
c
      xr2_inf   = 1.0d0 / (1.0d0 + dexp((V + 88.0d0) / 24.0d0))
      alpha_xr2 = 3.0d0 / (1.0d0 + dexp(((-60.0d0) - V) / 20.0d0))
      beta_xr2  = 1.12d0 / (1.0d0 + dexp((V - 60.0d0) / 20.0d0))
      tau_xr2   = 1.0d0 * alpha_xr2 * beta_xr2
c     
      xs_inf   = 1.0d0 / (1.0d0 + dexp(((-5.0d0) - V) / 14.0d0))
      alpha_xs = 1400.0d0 / dsqrt(1.0 + exp((5.0d0 - V) / 6.0d0))
      beta_xs  = 1.0d0 / (1.0d0 + dexp((V - 35.0d0) / 15.0d0))
      tau_xs   = (1.0d0 * alpha_xs * beta_xs) + 80.0d0
c      
      m_inf   = 1.0d0 / ((1.0d0 + dexp(((-56.86) - V) / 9.03d0) )**2)
      alpha_m = 1.0d0 / (1.0d0 + dexp(((-60.0d0) - V) / 5.0d0))
      beta_m  = (0.1d0 / (1.0d0 + dexp((V + 35.0d0) / 5.0d0))) +
     &          (0.1d0 / (1.0d0 + dexp((V - 50.0d0) / 200.0d0)))
      tau_m   = 1.0d0 * alpha_m * beta_m
c
      h_inf   = 1.0d0 / ((1.0d0 + dexp((V + 71.55d0) / 7.43d0))**2)
      if (V < (-40.0d0)) then
         alpha_h =  (0.057 * exp((-(V + 80.0)) / 6.8))
      else
         alpha_h = 0.0d0
      end if
      if ( V < (-40.0d0)) then
         beta_h = ((2.7d0*dexp(0.079d0*V))
     &          + (310000.0d0*dexp(0.3485d0*V)))
      else
         beta_h = (0.77d0/(0.13d0*(1.0d0+dexp((V+10.66d0)/(-11.1d0)))))
      end if
      tau_h   = 1.0d0 / (alpha_h + beta_h)
c      
      j_inf   = 1.0d0 / ((1.0d0 + exp((V + 71.55d0) / 7.43))**2)
      if ( V < (-40.0d0)) then
         alpha_j = ((((((-25428.0d0)*dexp(0.2444d0*V)) -
     &       (6.948e-06*dexp((-0.04391d0)*V)))*(V+37.78d0))/1.0d0) /
     &       (1.0d0+dexp(0.311d0*(V+79.23d0))))
      else
         alpha_j = 0.0d0
      end if
      if ( V < (-40.0d0) ) then
         beta_j = ((0.02424d0*dexp((-0.01052d0)*V)) /
     &            (1.0d0+dexp((-0.1378d0)*(V+40.14d0))))
      else
         beta_j =  ((0.6d0*dexp(0.057d0*V)) /
     &             (1.0d0+dexp((-0.1d0)*(V+32.0d0))))
      end if
      tau_j   = 1.0d0 / (alpha_j + beta_j)
c     
      d_inf = 1.0d0 / (1.0d0 + dexp(((-8.0d0) - V)/7.5d0))
      alpha_d = (1.4d0 / (1.0d0+dexp(((-35.0d0)-V)/13.0d0))) + 0.25d0
      beta_d  = 1.4d0 / (1.0d0+dexp((V+5.0d0)/5.0d0))
      gamma_d = 1.0d0 / (1.0d0+dexp((50.0-V)/20.0d0))
      tau_d   = (1.0d0*alpha_d*beta_d) + gamma_d
c
      f_inf = 1.0d0 / (1.0d0 + dexp((V + 20.0d0)/ 7.0d0))
      tau_f = (1102.5d0 * dexp((-((V + 27.0d0)**2)) / 225.0d0)) +
     &     (200.0d0/(1.0d0+dexp((13.0d0-V)/10.0d0))) +
     &     (180.0d0 / (1.0d0 + exp((V + 30.0d0) / 10.0d0))) + 20.0d0
c      
      f2_inf = (0.67d0 / (1.0d0 + dexp((V + 35.0d0) / 7.0d0))) + 0.33d0
      tau_f2 = (562.0d0 * dexp((-((V + 27.0d0)**2)) / 240.0d0)) +
     &     (31.0d0 / (1.0d0 + dexp((25.0d0 - V) / 10.0d0))) +
     &     (80.0d0 / (1.0d0 + dexp((V + 30.0d0) / 10.0d0)))
c     
      fCass_inf = (0.6d0 / (1.0d0 + ((Cass/0.05)**2))) + 0.4d0
      tau_fCass = (80.0d0 / (1.0d0 + ((Cass/0.05)**2))) + 2.0d0
c      
      s_inf = 1.0d0 / (1.0d0 + dexp((V + 20.0d0) / 5.0d0))
      tau_s = (85.0d0 * dexp((-((V + 45.0d0)**2)) / 320.0d0))
     &      + (5.0d0 / (1.0d0 + dexp((V - 20.0d0) / 5.0d0))) + 3.0d0
c      
      r_inf = 1.0d0 / (1.0d0 + dexp((20.0d0 - V) / 6.0d0))
      tau_r = (9.5d0 * dexp((-((V + 40.0d0)**2)) / 1800.0d0)) + 0.8d0
c      
      kcasr = max_sr-((max_sr-min_sr)/(1.0d0+((EC/CaSR)**2)))
      k1 = k1_prime / kcasr
      k3 = 0.06d0
      var_calcium_dynamics__o = (k1 * (Cass**2) * Rprime) /
     &                       (k3 + (k1 * (Cass**2)))
      Irel = V_rel * var_calcium_dynamics__o * (CaSR-Cass)
c
c     calcium dynamics stuff
c      
      var_Ca_K_up = 0.00025d0
      var_Ca_i_up = Vmax_up/(1.0d0+((var_Ca_K_up**2)/(Cai**2)))
      var_Ca_V_leak = 0.00036d0
      var_Ca_i_leak = var_Ca_V_leak*(CaSR-Cai)
c
      var_Ca_V_xfer = 0.0038d0
      var_Ca_i_xfer = var_Ca_V_xfer * (Cass-Cai)
      var_Ca_k2_prime = 0.045d0
      var_Ca_k2 = var_Ca_k2_prime * kcasr
c      
      var_Ca_k4 = 0.005d0
      var_Ca_Buf_c = 0.2d0
      var_Ca_K_buf_c = 0.001d0
      Ca_i_bufc = 1.0d0/(1.0d0+((var_Ca_Buf_c*var_Ca_K_buf_c) /
     &            ((Cai+var_Ca_K_buf_c)**2)))
      var_Ca_K_buf_sr = 0.3d0
      var_Ca_Buf_sr = 10.0d0
      var_Ca_Ca_sr_bufsr =1.0d0/(1.0d0+((var_Ca_Buf_sr*var_Ca_K_buf_sr)/
     &     ((CaSR+var_Ca_K_buf_sr)**2)))
      var_Ca_Buf_ss = 0.4d0
      var_Ca_K_buf_ss = 0.00025d0
      Ca_ss_bufss = 1.0d0/(1.0d0+((var_Ca_Buf_ss*var_Ca_K_buf_ss) /
     &     ((Cass+var_Ca_K_buf_ss)**2)))
      var_Ca_V_sr = 0.001094d0
      var_Ca_V_ss = 5.468e-05
c      
      var_Ca_V_c = V_c
      var_Ca_F = FF
      var_Ca_Cm = Cm
      var_Ca_ICaL = ICaL
      var_Ca_INaCa = INaCa
      var_Ca_IpCa = IpCa
      var_Ca_IbCa = IbCa
c
c     compute rates
c
      d_dt_V = -(IK1+Ito+IKr+IKs+ICaL+INaK+INa+IbNa+INaCa+IbCa+IpK+IpCa
     &         +xistim)
      d_dt_Xr1 = (xr1_inf - Xr1) / tau_xr1
      d_dt_Xr2 = (xr2_inf - Xr2) / tau_xr2
      d_dt_Xs = (xs_inf - Xs) / tau_xs
      d_dt_m = (m_inf - m) / tau_m
      d_dt_h = (h_inf - h) / tau_h
      d_dt_j = (j_inf - j) / tau_j
      d_dt_d = (d_inf - d) / tau_d
      d_dt_f = (f_inf - f) / tau_f
      d_dt_f2 = (f2_inf - f2) / tau_f2
      d_dt_fCass = (fCass_inf - fCass) / tau_fCass
      d_dt_s = (s_inf - s) / tau_s
      d_dt_r = (r_inf - r) / tau_r
      d_dt_Rprime = ((-var_Ca_k2) * Cass*Rprime) +
     &                (var_Ca_k4 * (1.0d0 - Rprime))
      d_dt_Cai = Ca_i_bufc*(((((var_Ca_i_leak-var_Ca_i_up)*var_Ca_V_sr)
     &     / var_Ca_V_c) + var_Ca_i_xfer) - ((((var_Ca_IbCa +
     &     var_Ca_IpCa) - (2.0d0 * var_Ca_INaCa)) * var_Ca_Cm) /
     &     (2.0d0 * var_Ca_V_c * var_Ca_F)))
      d_dt_CaSR = var_Ca_Ca_sr_bufsr*(var_Ca_i_up-(Irel+var_Ca_i_leak))
      d_dt_Cass = Ca_ss_bufss*(((((-var_Ca_ICaL)*var_Ca_Cm) /
     &     (2.0d0*var_Ca_V_ss*var_Ca_F))+((Irel*var_Ca_V_sr) /
     &      var_Ca_V_ss)) - ((var_Ca_i_xfer*var_Ca_V_c)/var_Ca_V_ss))
      d_dt_Nai = ((-(INa+IbNa+(3.0d0*INaK)+(3.0d0*INaCa)))/(V_c*FF))*Cm
      d_dt_Ki = ((-((IK1+Ito+IKr+IKs+IpK+xistim)-(2.0d0*INaK))) /
     &          (V_c*FF))*Cm
c     
      dsv( 1) = d_dt_V
      dsv( 2) = d_dt_Xr1 
      dsv( 3) = d_dt_Xr2
      dsv( 4) = d_dt_Xs
      dsv( 5) = d_dt_m
      dsv( 6) = d_dt_h
      dsv( 7) = d_dt_j
      dsv( 8) = d_dt_d
      dsv( 9) = d_dt_f
      dsv(10) = d_dt_f2
      dsv(11) = d_dt_fCass
      dsv(12) = d_dt_s
      dsv(13) = d_dt_r
      dsv(14) = d_dt_Cai;
      dsv(15) = d_dt_CaSR
      dsv(16) = d_dt_Cass
      dsv(17) = d_dt_Rprime
      dsv(18) = d_dt_Nai
      dsv(19) = d_dt_Ki
c
      return
      end
      
