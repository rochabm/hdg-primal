c ------------------------------------------------------------------------------
c     Modelo celular ten Tusscher 2006 - versao benchmark
c     Programa principal para teste
c ------------------------------------------------------------------------------            
      
      implicit none
c
      integer i,k,nstep,nsave
      integer iout,neq
      real*8 tf, dt, t, istim
      real*8 sv, dsv
c
      dimension sv(19), dsv(19)
c
      iout = 10
c
      open(unit=iout,  file= 'saida_cellmodel.dat')
c
c     configuracoes iniciais
c
      neq = 19
      tf = 600.0d0
      dt = 0.0001d0
      nstep = int(tf/dt)
      nsave = int(1.0d0/dt)
      write(*,*) tf, dt, nstep,nsave
c
c     condicao inicial
c     
      call tt2006_init(neq,sv)
      write(iout,"(3F16.8)") t, sv(1)
c
c     integracao no tempo
c     
      do k=1,nstep
         t = k*dt
c
c     estimulo
c
         if (t.gt.10.0d0.and.t.lt.11.0d0) then
            istim = -35.0d0
         else
            istim = 0.0d0
         end if
c
c     avanca edos
c        
         call tt2006_equation(neq,dt,sv,dsv,istim)
c
c     euler
c
         do i=1,neq
            sv(i) = sv(i) + dt*dsv(i)
         end do
c
c     escreve em arquivo
c     
         if(mod(k,nsave).eq.0) then
            write(*,*) t
            write(iout,"(3F16.8)") t, sv(1)
         end if
         
      end do     

c     
      close(iout)
c     
      end
