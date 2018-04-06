********************************************************************************
*              SUBROUTINES FOR FITZ HUGH NAGUMO  CELL MODEL                    *
********************************************************************************     
            
c ------------------------------------------------------------------------------      
      subroutine fhn_init(neq, sv)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 sv
      dimension sv(2)
c
      sv(1) = -1.19940803524d0
      sv(2) = -0.624260044055d0
c
      return
      end
      
c ------------------------------------------------------------------------------      
      subroutine fhn_equation(neq, dt, sv, dsv, istim)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 sv, dsv, dt, istim
c      
c     variaveis e constantes
c
      real*8 v,w,eps,gamma,beta,v3,dv,dw
c
      dimension sv(2), dsv(2)
c      
      eps = 0.2d0
      gamma = 0.8d0
      beta = 0.7d0
c      
      v = sv(1)
      w = sv(2)
c
c     calculos
c
      v3 = v**3
      dv = ((1.0d0/eps) * (v-((1.0d0/3.0d0)*v3)-w)) + istim
      dw = eps * (v - gamma*w + beta)
c
c     taxas
c      
      dsv(1) = dv
      dsv(2) = dw
c
      return
      end
      
