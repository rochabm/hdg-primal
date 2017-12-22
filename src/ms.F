********************************************************************************
*              SUBROUTINES FOR FITZ HUGH NAGUMO  CELL MODEL                    *
********************************************************************************     
            
c ------------------------------------------------------------------------------      
      subroutine ms_init(neq, sv)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 sv
      dimension sv(2)
c
      sv(1) = 0.0d0
      sv(2) = 1.0d0
c
      return
      end
      
c ------------------------------------------------------------------------------      
      subroutine ms_equation(neq, dt, sv, dsv, istim)
c ------------------------------------------------------------------------------      
      implicit none
c
      integer neq
      real*8 sv, dsv, dt, istim
c      
c     variaveis e constantes
c
      real*8 v,h,v_gate,tau_in,tau_out,tau_open,tau_closed
      real*8 jstim,jin,jout
      dimension sv(2), dsv(2)
c      
      v_gate     = 0.13d0
      tau_in     = 0.3d0  
      tau_out    = 6.0d0  
      tau_open   = 120.0d0
      tau_closed = 150.0d0
c      
      v = sv(1)
      h = sv(2)
c
c     correntes
c
      jstim = istim
      jin   = (h*(v*v*(1.0-v)))/tau_in
      jout  = -v/tau_out
c
c     taxas
c      
      dsv(1) = Jin + Jout + Jstim
      if ( v.lt.v_gate ) then
         dsv(2) = (1.0d0-h)/tau_open
      else
         dsv(2) = -h/tau_closed
      end if
c
      return
      end
      
