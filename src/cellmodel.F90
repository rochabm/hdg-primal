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
ccc   dimension sv(2), dsv(2)
ccc   dimension sv(19), dsv(19)
      dimension sv(12), dsv(12)
c
      iout = 10
c
      open(unit=iout,  file= 'saida_cellmodel.dat')
c
c     configuracoes iniciais
c
      
ccc   neq = 19
ccc   neq = 2
      neq = 12
      
      tf = 600.0d0
      dt = 0.02d0
      nstep = int(tf/dt)
      nsave = int(1.0d0/dt)
      write(*,*) tf, dt, nstep,nsave
c
c     condicao inicial
c
      call tt3_init(neq,sv)
ccc      call tt2006_init(neq,sv)
ccc      call ms_init(neq,sv)
      write(iout,"(3F16.8)") t, sv(1)
c
c     integracao no tempo
c     
      do k=1,nstep
         t = k*dt
c         write(*,*) "tempo ", t
c
c     estimulo
c
         istim = 0.d0
c         
         if (t.gt.100.0d0.and.t.lt.101.0d0) then
            istim = -52.0d0
c            istim = -35.0d0
c           istim = 0.2d0
         end if
c
c     avanca edos
c
         call tt3_equation(neq,dt,sv,dsv,istim)
ccc         call tt2006_equation(neq,dt,sv,dsv,istim)
ccc         call ms_equation(neq,dt,sv,dsv,istim)
c
c     euler
c
         do i=1,neq
ccc            sv(i) = sv(i) + dt*dsv(i)

!     tt2006
ccc            if(i.ge.2.and.i.le.13) then
ccc               sv(i) = dsv(i)
ccc            else
ccc               sv(i) = sv(i) + dt*dsv(i)
ccc            end if 

!     tt3
            if(i.ge.2.and.i.le.12) then
               sv(i) = dsv(i)
            else
               sv(i) = sv(i) + dt*dsv(i)
            end if
                        
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
