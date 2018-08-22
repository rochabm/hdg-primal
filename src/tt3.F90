********************************************************************************
*              SUBROUTINES FOR TEN TUSSCHER 3 CELL MODEL                       *
********************************************************************************     
            
c ------------------------------------------------------------------------------      
      subroutine tt3_init(neq, sv)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq, i
      real*8 sv
      dimension sv(12)
c
      sv( 1) = -86.2d0   ! Vm
      sv( 2) = 0.0d0     ! m
      sv( 3) = 0.75d0    ! h
      sv( 4) = 0.75d0    ! j
      sv( 5) = 0.0d0     ! xr1
      sv( 6) = 0.0d0     ! xs
      sv( 7) = 1.0d0     ! s 
      sv( 8) = 1.0d0     ! f
      sv( 9) = 1.0d0     ! f2
c
      sv(10) = 0.0d0     ! d_inf 
      sv(11) = 0.0d0     ! r_inf 
      sv(12) = 0.0d0     ! xr2_inf 
c
      return
      end
      
c ------------------------------------------------------------------------------      
      subroutine tt3_equation(neq, dt, sv, dsv, istim)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 dt, istim, sItot, acid_voltage
c      
c     variaveis
c      
      real*8 svolt,sm,sh,sj,sxr1,sxs,ss,sf,sf2
      real*8 D_INF,R_INF,XR2_INF
c
c     derivadas
c
      real*8 d_dt_svolt
      real*8 d_dt_sm
      real*8 d_dt_sh
      real*8 d_dt_sj
      real*8 d_dt_sxr1
      real*8 d_dt_sxs
      real*8 d_dt_ss
      real*8 d_dt_sf
      real*8 d_dt_sf2

!-----------------------------------------------------------------------------
!  ELECTROPHYSIOLOGICAL PARAMETERS:
!-----------------------------------------------------------------------------*/

!     Concentrations
      real*8 Ko,Cao,Nao
      real*8 Cai, Nai, Ki

!     ATP
      real*8 natp, nicholsarea
      real*8 atpi,hatp,katp,patp,gkatp,gkbaratp
     
!     constants
      real*8 R, F, T, RTONF

!     Cellular capacitance         
      real*8 CAPACITANCE
      
c
c     Parameters for currents
c
      
!     Parameters for IKr
      real*8 Gkr
!     Parameters for Iks
      real*8 pKNa
      real*8 Gks
!     Parameters for Ik1
      real*8 GK1
!     Parameters for Ito
      real*8 Gto
!     Parameters for INa
      real*8 GNa      
!     Parameters for IbNa
      real*8 GbNa      
!     Parameters for INaK
      real*8 KmK
      real*8 KmNa
      real*8 knak      
!     Parameters for ICaL
      real*8 katp2, hatp2, pcal, GCaL
!     Parameters for IbCa
      real*8 GbCa      
!     Parameters for INaCa
      real*8 knaca, KmNai, KmCa, ksat, n      
!     Parameters for IpCa
      real*8 GpCa, KpCa      
!     Parameters for IpK
      real*8 GpK

!     Outros      
      real*8 Ek,Ena,Eks,Eca
      real*8 IKr,IKs,IK1,Ito,INa,IbNa,ICaL,IbCa,INaCa,IpCa,IpK,INaK
      real*8 IKatp

      real*8 Ak1,Bk1
      real*8 rec_iK1
      real*8 rec_ipK
      real*8 rec_iNaK
      real*8 AM,BM
      real*8 AH_1,BH_1,AH_2,BH_2
      real*8 AJ_1,BJ_1,AJ_2,BJ_2
      real*8 M_INF,H_INF,J_INF
      real*8 TAU_M,TAU_H,TAU_J
      real*8 axr1,bxr1,axr2,bxr2
      real*8 Xr1_INF,TAU_Xr1
      real*8 Axs,Bxs
      real*8 Xs_INF,TAU_Xs
      real*8 S_INF,TAU_S
      real*8 Ad,Bd,Cd,Af,Bf,Cf
      real*8 Af2,Bf2,Cf2
      real*8 TAU_F,F_INF
      real*8 TAU_F2,F2_INF      
      real*8 D_INF_NEW
      real*8 Xr2_INF_NEW
      real*8 R_INF_NEW      
c
c     vetores
c      
      real*8 sv,dsv     
      dimension sv(12), dsv(12)
c
c     recupera variaveis
c     Vm,m,h,   j,xr1,xs,   s,f,f2     
      svolt = sv(1)
      sm = sv(2)
      sh = sv(3)   
      sj = sv(4)
      sxr1 = sv(5)      
      sxs = sv(6)     
      ss = sv(7)     
      sf = sv(8)      
      sf2 = sv(9)

      d_inf = sv(10)
      r_inf = sv(11)
      xr2_inf = sv(12)
c
c     valores
c      
      Ko = 5.4d0

      natp = 0.24d0           ! K dependence of ATP-sensitive K current
      nicholsarea = 0.00005d0 ! Nichol's areas (cm^2)
      atpi = 4.0d0            ! Intracellular ATP concentraion (mM)
      hatp = 2.0d0            ! Hill coefficient
      katp = -0.0942857142857d0*atpi + 0.683142857143d0 
      patp =  1.0d0/(1.0d0 + (atpi/katp)**hatp)
      gkatp = 0.000195d0/nicholsarea
      gkbaratp =  gkatp*patp*((Ko/4.0d0)**natp)

      Cao = 2.0d0
      Nao = 140.0d0
      Cai = 0.00007d0
      Nai = 7.67d0
      Ki = 138.3d0

!     constants
      R=8314.472d0
      F=96485.3415d0
      T=310.0d0
      RTONF=(R*T)/F

!     Cellular capacitance         
      CAPACITANCE=0.185d0

!     Parameters for currents
!     Parameters for IKr
      Gkr=0.101d0
!     Parameters for Iks
      pKNa=0.03d0
      
!#ifdef EPI
!      Gks=0.257d0
!#ifdef ENDO
      Gks=0.392;
!#ifdef MCELL
!real*8 Gks=0.098;

!     Parameters for Ik1
      GK1=5.405d0
      
!     Parameters for Ito
!#ifdef EPI
!      Gto=0.294d0
!#ifdef ENDO
      Gto = 0.073d0
!#ifdef MCELL
!real*8 Gto=0.294;

!     Parameters for INa
      GNa=14.838d0
      
!     Parameters for IbNa
      GbNa=0.00029d0
      
!     Parameters for INaK
      KmK=1.0d0
      KmNa=40.0d0
      knak=2.724d0
      
!     Parameters for ICaL
      katp2= 1.4d0
      hatp2 = 2.6d0
      pcal = 1.0d0/(1.0d0 + ((katp2/atpi)**hatp2))
      GCaL = 0.2786d0*pcal  !Multiply by 0.88

!     Parameters for IbCa
      GbCa=0.000592d0
      
!     Parameters for INaCa
      knaca=1000d0
      KmNai=87.5d0
      KmCa=1.38d0
      ksat=0.1d0
      n=0.35d0
      
!     Parameters for IpCa
      GpCa=0.1238d0
      KpCa=0.0005d0
      
!     Parameters for IpK
      GpK=0.0293d0
c
c     begin computations
c      
      Ek = RTONF*(dlog((Ko/Ki)))
      Ena = RTONF*(dlog((Nao/Nai)))
      Eks = RTONF*(dlog((Ko+pKNa*Nao)/(Ki+pKNa*Nai)))
      Eca = 0.5d0*RTONF*(dlog((Cao/Cai)))      
      
!     Needed to compute currents
      Ak1 = 0.1d0/(1.0d0 + dexp(0.06d0*(svolt-Ek-200.0d0)));
      Bk1 = (3.0d0*dexp(0.0002d0*(svolt-Ek+100.0d0))+
     &    dexp(0.1d0*(svolt-Ek-10.0d0)))/(1.0d0+dexp(-0.5d0*(svolt-Ek)))
      rec_iK1 = Ak1/(Ak1+Bk1)
      rec_iNaK = (1.0d0/(1.0d0+0.1245d0*
     &    dexp(-0.1d0*svolt*F/(R*T))+0.0353d0*dexp(-svolt*F/(R*T))))
      rec_ipK = 1.0d0/(1.0d0+dexp((25.0d0-svolt)/5.98d0));


!     Compute currents
      INa = GNa*sm*sm*sm*sh*sj*(svolt-Ena)
      ICaL = GCaL*D_INF*sf*sf2*(svolt-60.0d0)
      Ito = Gto*R_INF*ss*(svolt-Ek)
      IKr = Gkr*dsqrt(Ko/5.4d0)*sxr1*Xr2_INF*(svolt-Ek)
      IKs = Gks*sxs*sxs*(svolt-Eks)
      IK1 = GK1*rec_iK1*(svolt-Ek)
      INaCa = knaca*
     &     (1./(KmNai*KmNai*KmNai+Nao*Nao*Nao))*(1.0d0/(KmCa+Cao))*
     &     (1.0d0/(1.0d0+ksat*dexp((n-1.0d0)*svolt*F/(R*T))))*
     &     (dexp(n*svolt*F/(R*T))*Nai*Nai*Nai*Cao-
     &     dexp((n-1.0d0)*svolt*F/(R*T))*Nao*Nao*Nao*Cai*2.5d0)      
      INaK = knak*(Ko/(Ko+KmK))*(Nai/(Nai+KmNa))*rec_iNaK
      IpCa = GpCa*Cai/(KpCa+Cai)
      IpK = GpK*rec_ipK*(svolt-Ek)
      IbNa = GbNa*(svolt-Ena)
      IbCa = GbCa*(svolt-Eca)

      IKatp = gkbaratp*(svolt-Ek)

c
c     stimulus
c
      if(istim.ne.0) then
         write(*,*) istim
!         stop
      endif

!     Determine total current
      sItot = IKr + IKs + IK1 + Ito + INa + IbNa + ICaL + IbCa
     &      + INaK + INaCa + IpCa + IpK + IKatp + Istim
      
!     compute steady state values and time constants 

      acid_voltage = svolt - 3.4d0
c
c     state variable m
c      
      AM = 1.0d0/(1.0d0+dexp((-60.0d0-acid_voltage)/5.0d0))
      BM = 0.1d0/(1.0d0+dexp((acid_voltage+35.0d0)/5.0d0))+
     &     0.10d0/(1.0d0+dexp((acid_voltage-50.0d0)/200.0d0))
      TAU_M = AM*BM
      M_INF = 1.0d0/((1.0d0+dexp((-56.86d0-acid_voltage)/9.03d0))*
     &     (1.0d0+dexp((-56.86d0-acid_voltage)/9.03d0)))
c
c     state variable h
c      
      if (svolt.ge.-40.0d0) then
      AH_1 = 0.0d0 
      BH_1 = (0.77d0/(0.13d0*
     &      (1.0d0+dexp(-(acid_voltage+10.66d0)/11.1d0))))
      TAU_H = 1.0d0/(AH_1+BH_1)
      else 
         AH_2 = (0.057d0*dexp(-(acid_voltage+80.0d0)/6.8d0))
         BH_2 = (2.7d0*dexp(0.079d0*acid_voltage)+(3.1e5)*
     &        dexp(0.3485d0*acid_voltage))
         TAU_H = 1.0d0/(AH_2+BH_2)
      end if
      H_INF = 1.0d0/((1.0d0+dexp((acid_voltage+71.55d0)/7.43d0))*
     &     (1.0d0+dexp((acid_voltage+71.55d0)/7.43d0)))
c
c     state variable j
c      
      if (svolt.ge.-40.0d0) then
         AJ_1 = 0.0d0
         BJ_1 = (0.6d0*dexp((0.057d0)*acid_voltage)/(1.0d0+
     &      dexp(-0.1d0*(acid_voltage+32.0d0))))
         TAU_J = 1.0d0/(AJ_1+BJ_1)         
      else    
         AJ_2 = (((-2.5428e4)*dexp(0.2444d0*acid_voltage)-(6.948e-6)*
     &        dexp(-0.04391d0*acid_voltage))*(acid_voltage+37.78d0)/
     &        (1.0d0+dexp(0.311d0*(acid_voltage+79.23d0))))
         BJ_2 = (0.02424d0*dexp(-0.01052d0*acid_voltage)/(1.0d0+
     &        dexp(-0.1378d0*(acid_voltage+40.14d0))))
         TAU_J = 1.0d0/(AJ_2+BJ_2)
      end if
      J_INF = H_INF
c
c     state variable xr1
c      
      Xr1_INF = 1.0d0/(1.0d0+dexp((-26.0d0-svolt)/7.0d0))
      axr1 = 450.0d0/(1.0d0+dexp((-45.0d0-svolt)/10.0d0))
      bxr1 = 6.0d0/(1.0d0+dexp((svolt-(-30.0d0))/11.5d0))
      TAU_Xr1 = axr1*bxr1
      
      !Xr2_INF = 1.0d0/(1.0d0+dexp((svolt-(-88.0d0))/24.0d0))
      Xr2_INF_NEW = 1.0d0/(1.0d0+dexp((svolt-(-88.0d0))/24.0d0))
      
      Xs_INF = 1.0d0/(1.0d0+dexp((-5.0d0-svolt)/14.0d0))
      Axs=(1400.0d0/(dsqrt(1.0d0+dexp((5.0d0-svolt)/6.0d0))))
      Bxs=(1.0d0/(1.0d0+dexp((svolt-35.0d0)/15.0d0)))
      TAU_Xs = Axs*Bxs + 80.0d0

!     EPI
!      R_INF = 1.0d0/(1.0d0+dexp((20.0d0-svolt)/6.0d0))
!      S_INF = 1.0d0/(1.0d0+dexp((svolt+20.0d0)/5.0d0))
!      TAU_S = 85.0d0*dexp(-(svolt+45.0d0)*(svolt+45.0d0)/320.0d0)+
!     &    5.0d0/(1.0d0+dexp((svolt-20.0d0)/5.0d0))+3.0d0

!     ENDO
      R_INF_NEW = 1.0d0/(1.0d0+dexp((20.0d0-svolt)/6.0d0))
      S_INF = 1.0d0/(1.0d0+dexp((svolt+28)/5.0d0))
      TAU_S = 1000.0d0*dexp(-(svolt+67.0d0)*
     &        (svolt+67.0d0)/1000.0d0)+8.0d0

!     MCELL
      !R_INF=1./(1.+exp((20-svolt)/6.));
      !S_INF=1./(1.+exp((svolt+20)/5.));
      !TAU_S=85.*exp(-(svolt+45.)*(svolt+45.)/320.)+5./(1.+exp((svolt-20.)/5.))+3.;

      !D_INF =1.0d0/(1.0d0+dexp((-8.0d0-svolt)/7.5d0))
      D_INF_NEW =1.0d0/(1.0d0+dexp((-8.0d0-svolt)/7.5d0))
      
      F_INF =1.0d0/(1.0d0+dexp((svolt+20.0d0)/7.0d0))
      Af = 1102.5d0*dexp(-(svolt+27.0d0)*(svolt+27.0d0)/225.0d0)
      Bf = 200.0d0/(1.0d0+dexp((13.0d0-svolt)/10.0d0))
      Cf = (180.0d0/(1.0d0+dexp((svolt+30.0d0)/10.0d0)))+20.0d0
      TAU_F = Af + Bf + Cf
      
      F2_INF = 0.67d0/(1.0d0+dexp((svolt+35.0d0)/7.0d0))+0.33d0
      Af2 = 600.0d0*dexp(-(svolt+27.0d0)*(svolt+27.0d0)/170.0d0)
      Bf2 = 7.75d0/(1.0d0+exp((25.0d0-svolt)/10.0d0))
      Cf2 = 16.0d0/(1.0d0+exp((svolt+30.0d0)/10.0d0))
      TAU_F2 = Af2 + Bf2 + Cf2
      
      !write(*,*) D_INF_NEW, R_INF_NEW, XR2_INF_NEW
c
c     Compute RHS
c     
      dsv( 1) = -sItot ! d_dt_v
c
c     HH variables -> Rush-Larsen scheme
c      
      dsv(2) = m_inf   - (m_inf-sm)    *dexp(-dt/tau_m)
      dsv(3) = h_inf   - (h_inf-sh)    *dexp(-dt/tau_h)
      dsv(4) = j_inf   - (j_inf-sj)    *dexp(-dt/tau_j)
      dsv(5) = xr1_inf - (xr1_inf-sxr1)*dexp(-dt/tau_xr1)
      dsv(6) = xs_inf  - (xs_inf-sxs)  *dexp(-dt/tau_xs)     
      dsv(7) = s_inf   - (s_inf-ss)    *dexp(-dt/tau_s)
      dsv(8) = f_inf   - (f_inf-sf)    *dexp(-dt/tau_f)
      dsv(9) = f2_inf  - (f2_inf-sf2)  *dexp(-dt/tau_f2)
c
      dsv(10) = D_INF_NEW
      dsv(11) = R_INF_NEW
      dsv(12) = Xr2_INF_NEW
c      
      return
      end
