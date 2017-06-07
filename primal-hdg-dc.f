c     ************************************************************
c     formulacao hibrida primal - Problema de Poisson/Convec-Difus
c     Multplicador = potencial na interface dos elementos
c
c     ************************************************************
c     *                                                          *
c     *            * * *   Poisson Problem      * * *            *
c     *                                                          *
c     *                                                          *
c     *         FORMULACAO HIBRIDA PRIMAL ESTABILIZADA           *
c     *                                                          *
c     *                 MULTIPLICADOR = POTENCIAL                *
c     *                                                          *
c     *                                                          *
c     *                    ABIMAEL LOULA                         *
c     *                                                          *
c     *                   Setembro de 2016                       *
c     ************************************************************

c     Residuo no interior dos elementos:
c     (grad u , grad v ) + (grad u,v) - (f,v)
c
c     Termos de saltos nas arestas:
c     beta(u - up, v-vp)
c     -(grad(u).n , (v-vp))
c     lambda((u-up) , grad(v).n)
c
c     program to set storage capacity, precision and input/output units
c
      common /bpoint/ mfirst,mlast,ilast,mtot,iprec
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      character*4 ia
      parameter (ndim=200000000)

c     for 3d problem with 16x16x16 elements this does not work
c     parameter (ndim=200000000)
c     ok
c     parameter (ndim=4000000000)
c
      common a(ndim)
      common /dictn/ ia(10000000)
c     old
c      common /dictn/ ia(100000000)

c
c     mfirst = address of first available word in blank common
c     mlast  = address of last available word in blank common
c     mtot   = total storage allocated to blank common
c     iin    = input unit number
c     ieco   = output unit of input data
c     ilocal = output unit of post-processed displacements      
c     iprec  = precision flag; eq.1, single precision
c                              eq.2, double precision
c
      iin    = 8
      ipp    = 110
      ipmx   = 11
      ieco   = 12
      ilp    = 130
      ilocal = 13
      interpl= 14
      ielmat = 15
c
      open(unit=iin,     file= 'primal-hdg-dc.dat',status='old')
      open(unit=ieco,    file= 'saida-hdg-dc.eco')
      open(unit=ipp,     file= 'errofem-primal-dc-shdg.dat')
      open(unit=ilp,     file= 'erro-local-primal-shdg-dc.dat')
      open(unit=interpl, file= 'erro-interpolante.dat')
      open(unit=ielmat,  file= 'matriz-elemento-primal-shdg-dc.dat')
c
c     open(unit=ilocal, file= 'erro-local-pp-misto-hdg.con')
c     open(unit=ipmx, file= 'errofem-pp-misto-hdg.con')
c
      mfirst = 1
      ilast  = 0
      mlast  = ndim
      mtot   = ndim
      iprec  = 2      
c
c     main subroutine
c
      call lpgm
c
c     system-dependent unit/file specifications
c
      close(iin)
      close(ieco)
      close(ipmx)
      close(ipp)
      close(ipl)
      close(ilocal)
      close(interpl)
      close(ielmat)
c
      stop
      end
c
c----------------------------------------------------------------------
      subroutine lpgm
c----------------------------------------------------------------------
c     LPGM - a linear static finite element analysis program for
c     Petrov Galerkin methods : global driver
c---------------------------------------------------------------------- 
      real*8 zero,pt1667,pt25,pt5,one,two,three,four,five,six,tempf
      character*4 title,titlea(20)
c
c     remove above card for single-precision operation
c
c     catalog of common statements
c
      common /bpoint/ mfirst,mlast,ilast,mtot,iprec
      common /colhtc/ neq
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /etimec/ etime(6)
      common /genelc/ n,nel(3),incel(3),inc(3)
      common /genflc/ tempf(6,20),nf,numgpf,nincf(3),incf(3)
      common /info  / iexec,iprtin,irank,nsd,numnp,ndof,nlvect,
     &                numeg,nmultp,nedge
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /labels/ labeld(3),label1(16),label2(3)
      common /spoint/ mpd,mpx,mpid,mpf,mpdiag,mpngrp,
     &                mpalhs,mpdlhs,mpbrhs,mped,index
      common /titlec/ title(20)
      character*4 ia
      common a(1)
      common /dictn/ ia(10000000)
c
c     input phase
c
      write(*,'(A)') "subroutine echo"
      call echo
c
  100 continue
      do 200 i=1,6
  200 etime(i) = 0.0
      call timing(t1)
      titlea = title
      read(iin,1000) title
      if (title(1).eq.'*end') then
      write(ipp,11)
  11  format(/
     $         6x,'-log(h)',7x,'log(k)',
     $         7x,'log(p)',2x,'log(grad p)', 4x,'log(mult)',//,
     $ 10x,'Estudo de convergencia das aproximacoes hibridas',/,
     & 10x,'primal-pp-primal: (p, gard p, mult)',/)
      write(ipp,3333) titlea
      write(ipmx,10)
  10  format(/
     $         6x,'-log(h)',7x,'log(k)',7x,'log(u)',2x,'log(grad u)',
     $         7x,'log(p)',2x,'log(grad p)', 4x,'log(mult)',
     $         2x,'conservacao',//,
     $ 10x,'Estudo de convergencia das aproximacoes hibridas',/,
     & 10x,'primal-pp-primal: (p, gard p, mult)',/,
     & 10x,'primal-pp-mista estabilizada: (u, gard u)',/)
      write(ipmx,3333) titlea
3333  format(/,20a4)
c
      write(ilp,21)
  21  format(/
     $         6x,'-log(h)',7x,'log(k)',
     $         7x,'log(p)',2x,'log(grad p)', 4x,'log(mult)',//,
     $ 10x,'Estudo de convergencia das projecoes locais',/,
     & 10x,'primal-pp-primal: (p, gard p, mult)',/)
      write(ilp,3333) titlea
c
      write(ilocal,20)
  20  format(/
     $         6x,'-log(h)',7x,'log(k)',7x,'log(u)',2x,'log(grad u)',
     $         7x,'log(p)',2x,'log(grad p)',
     $         2x,'conservacao',//,
     $ 10x,'Estudo de convergencia das projecoes locais',/,
     & 10x,'primal-pp-primal: (p, gard p)',/,
     & 10x,'primal-pp-mista estabilizada: (u, gard u)',/)
      write(ilocal,3333) titlea
c
      write(interpl,30)
  30  format(/
     $         6x,'-log(h)',7x,'log(k)',7x,'log(u)',2x,'log(grad u)',
     $         7x,'log(p)',2x,'log(grad p)', 4x,'log(mult)',
     $         2x,'conservacoo',//,
     $ 10x,'Estudo de convergencia das interpolantes',/)
      write(interpl,3333) titlea
c

      return
      end if
      read(iin,2000) iexec,iprtin,irank,
     &           nsd,numnp,ndof,nlvect,numeg,nedge,npar
c     *** DEBUG ***
      write(*,"(A,10I8)") "firstline ", iexec,iprtin,irank,nsd,numnp,
     &           ndof,nlvect,numeg,nedge,npar
c     *** DEBUG ***
      write(ieco,3000) title, iexec,iprtin
      write(ieco,4000) irank, nsd, numnp, ndof,
     &                 nlvect,numeg,nedge,npar
c
       nmultp = nedge*npar
c
c     initialization phase
c     set memory pointers for static data arrays,
c     and call associated input routines
c
      mpd    = mpoint('d       ',ndof  ,nmultp ,0 ,iprec)
      mpx    = mpoint('x       ',nsd   ,numnp  ,0 ,iprec)
      mped   = mpoint('ideg    ',2*ndof,nedge  ,0 ,1)
      mpid   = mpoint('id      ',ndof  ,nmultp ,0 ,1)
c
      if (nlvect.eq.0) then
         mpf = 1
      else
         mpf = mpoint('f       ',ndof  ,nmultp ,nlvect,iprec)
      endif
c
c     input coordinate data
c
      call coord(a(mpx),nsd,numnp,iprtin)
c
c     input boundary condition data and establish equation numbers
c
      call bcedge(a(mped),a(mpid),npar,nedge,ndof,nmultp,neq,iprtin)
c
c     input nodal force and prescribed kinematic boundary-value data
c
      if (nlvect.gt.0) call input(a(mpf),ndof,nmultp,0,nlvect,
     &                            iprtin)
c
c     allocate memory for idiag array and clear
c
      mpdiag = mpoint('idiag   ',neq   ,0     ,0     ,1)
      call iclear(a(mpdiag),neq)
c
      mpngrp = mpoint('ngrp    ',numeg ,0     ,0     ,1)
c
c     input element data
c
      call elemnt('input___',a(mpngrp))
      call timing(t2)
      etime(1) = t2 - t1
c
c     determine addresses of diagonals in left-hand-side matrix
c
      call diag(a(mpdiag),neq,nalhs)
c
c     allocate memory for global equation system
c
      mpalhs = mpoint('alhs    ',nalhs,0,0,iprec)
      mpdlhs = mpoint('dlhs    ',nalhs,0,0,iprec)
      mpbrhs = mpoint('brhs    ',neq  ,0,0,iprec)
      meanbw = nalhs/neq
      nwords = mtot - mlast + mfirst - 1

      write(*,*) "NWORDS", nwords
      write(*,*) "MTOT", mtot
      write(*,*) "MLAST",mlast
c
c     write equation system data
c
      write(ieco,5000) title,neq,nalhs,meanbw,nwords
c
c     solution phase
c
      write(*,'(A)') "subroutine driver"
      if (iexec.eq.1) call driver(neq,nalhs)
c
c     print memory-pointer dictionary
c
c      call prtdc
c
      call timing(t1)
      etime(2) = t1 - t2
c
c     print elapsed time summary
c
      call timlog
      go to 100
c
 1000 format(20a4)
 2000 format(16i10)
 3000 format(///,20a4///
     &' e x e c u t i o n   c o n t r o l   i n f o r m a t i o n '//5x,
     &' execution code  . . . . . . . . . . . . . .(iexec ) = ',i10//5x,
     &'    eq. 0, data check                                   ',   /5x,
     &'    eq. 1, execution                                    ',  //5x,
     &' input data print code . . . . . . . . . . .(iprtin) = ',i10//5x,
     &'    eq. 0, print nodal and element input data           ',   /5x,
     &'    eq. 1, do not print nodal and element input data    ',   /5x)
 4000 format(5x,
     &' rank check code . . . . . . . . . . . . .. (irank ) = ',i10//5x,
     &'    eq. 0, do not perform rank check                    ',   /5x,
     &'    eq. 1, print numbers of zero and nonpositive pivots ',   /5x,
     &'    eq. 2, print all pivots                             ',  //5x,
     &' number of space dimensions  . . . . . . . .(nsd   ) = ',i10//5x,
     &' number of nodal points  . . . . . . . . . .(numnp ) = ',i10//5x,
     &' number of nodal degrees-of-freedom  . . . .(ndof  ) = ',i10//5x,
     &' number of load vectors  . . . . . . . . . .(nlvect) = ',i10//5x,
     &' number of element groups  . . . . . . . . .(numeg ) = ',i10//5x,
     &' number of edge. . . .  .  . . . . . . . . .(nedges) = ',i10//5x)
 5000 format(///,20a4///
     &' e q u a t i o n    s y s t e m    d a t a              ',  //5x,
     &' number of equations . . . . . . . . . . . . (neq   ) = ',i8//5x,
     &' number of terms in left-hand-side matrix  . (nalhs ) = ',i8//5x,
     &' mean half bandwidth . . . . . . . . . . . . (meanbw) = ',i8//5x,
     &' total length of blank common required . . . (nwords) = ',i8    )
c
      end

c-----------------------------------------------------------------------
      subroutine driver(neq,nalhs)
c-----------------------------------------------------------------------
c     solution driver program
c-----------------------------------------------------------------------
      common /etimec/ etime(6)
      common /info  / iexec,iprtin,irank,nsd,numnp,ndof,nlvect,
     &                numeg,nmultp,nedge
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /spoint/ mpd,mpx,mpid,mpf,mpdiag,mpngrp,
     &                mpalhs,mpdlhs,mpbrhs,mped,index
      character*4 ia
      common a(1)
      common /dictn/ ia(10000000)
c
c     clear left and right hand side
c
      call clear(a(mpalhs),nalhs)
      call clear(a(mpdlhs),nalhs)
      call clear(a(mpbrhs),neq)
c
      call elemnt('form_stb',a(mpngrp))
c
c      account the nodal forces in the r.h.s.
c
      if (nlvect.gt.0)
     &   call load(a(mpid),a(mpf),a(mpbrhs),ndof,nmultp,nlvect)
c
c     clear displacement array
c
      call clear(a(mpd),ndof*nmultp)
c
      if (nlvect.gt.0)
     &   call ftod(a(mpid),a(mpd),a(mpf),ndof,nmultp,nlvect)
c
c     form the l.h.s and r.h.s. at element level
c
      call elemnt('form_lrs',a(mpngrp))
c
c     factorization of the stiffnes matrix
c        factor: symmetric matrices
c        factorns: for non-symmetric matrices
c
      write(*,*) "solving global system"
      write(*,*) "factorization of stiff matrix"
c     
      if(neq.eq.0) go to 1111
c     
      call factor(a(mpalhs),a(mpdiag),neq)
c
c     back substitution
c        back: for symmetric matrices
c        backns: for non-symmetric matrices      
c
      write(*,*) "back substitution"
      call back(a(mpalhs),a(mpbrhs),a(mpdiag),neq)
c
 1111 continue
c
      call btod(a(mpid),a(mpd),a(mpbrhs),ndof,nmultp)
c
c     write output
c
      call printd(' d i s p l a c e m e n t s                  ',
     &            a(mpd),ndof,nmultp,ieco)
c
c     post-processing phase
c
      call elemnt('pos_proc',a(mpngrp))
c
c100   continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine addlhs(alhs,eleffm,idiag,lm,nee,diag)
c-----------------------------------------------------------------------
c     program to add element left-hand-side matrix to
c        global left-hand-side matrix
c        diag = .true., add diagonal element matrix
c        diag = .false, add upper triangle of full element matrix
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c     
      logical diag
      dimension alhs(*),eleffm(nee,*),idiag(*),lm(*)
c     
      if (diag) then
c     
         do j=1,nee
            k = lm(j)
            if (k.gt.0) then
               l = idiag(k)
               alhs(l) = alhs(l) + eleffm(j,j)
            endif
         end do
c     
      else
c     
         do j=1,nee
            k = lm(j)
            if (k.gt.0) then
c     
               do i=1,j
                  m = lm(i)
                  if (m.gt.0) then
                     if (k.ge.m) then
                        l = idiag(k) - k + m
                     else
                        l = idiag(m) - m + k
                     endif
                     alhs(l) = alhs(l) + eleffm(i,j)
                  endif
               end do
c     
            endif
         end do
c     
      endif
c     
      return
      end

c-----------------------------------------------------------------------
      subroutine addrhs(brhs,elresf,lm,nee)
c-----------------------------------------------------------------------
c     program to add element residual-force vector to
c     global right-hand-side vector
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension brhs(*),elresf(*),lm(*)
c
      do j=1,nee
         k = lm(j)
         if (k.gt.0) brhs(k) = brhs(k) + elresf(j)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine back(a,b,idiag,neq)
c-----------------------------------------------------------------------
c     program to perform forward reduction and back substitution
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c     
      dimension a(*),b(*),idiag(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c     
c     forward reduction
c     
      jj = 0
c     
      do 100 j=1,neq
         jjlast = jj
         jj     = idiag(j)
         jcolht = jj - jjlast
         if (jcolht.gt.1)
     &        b(j) = b(j) - coldot(a(jjlast+1),b(j-jcolht+1),jcolht-1)
 100  continue
c     
c     diagonal scaling
c     
      do 200 j=1,neq
         ajj = a(idiag(j))
         if (ajj.ne.zero) b(j) = b(j)/ajj
 200  continue
c     
c     back substitution
c     
      if (neq.eq.1) return
      jjnext = idiag(neq)
c     
      do 400 j=neq,2,-1
         jj     = jjnext
         jjnext = idiag(j-1)
         jcolht = jj - jjnext
         if (jcolht.gt.1) then
            bj = b(j)
            istart = j - jcolht + 1
            jtemp  = jjnext - istart + 1
c     
            do 300 i=istart,j-1
               b(i) = b(i) - a(jtemp+i)*bj
 300        continue
c     
         endif
c     
 400  continue
c     
      return
      end

c-----------------------------------------------------------------------
      subroutine bcedge(ideg,id,npar,nedge,ndof,numnp,neq,iprtin)
c-----------------------------------------------------------------------
c     program to read, generate and write boundary condition data
c        and establish equation numbers
c-----------------------------------------------------------------------
      dimension id(ndof,*),ideg(2*ndof,*)
c
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      logical pflag
c
      call iclear(ideg,2*ndof*nedge)
      call iclear(id,ndof*numnp)
      call igen(ideg,2*ndof)
c
      if (iprtin.eq.0) then
         nn=0
         do 200 n=1,nedge
         pflag = .false.
c
         do 100 i=1,2*ndof
         if (ideg(i,n).ne.0) pflag = .true.
  100    continue
c
         if (pflag) then
            nn = nn + 1
            if (mod(nn,50).eq.1) write(ieco,1000) (i,i=1,2*ndof)
            write(ieco,2000) n,(ideg(i,n),i=1,2*ndof)
         endif
  200    continue
      endif
c
c    id - prescribed dof
c
        kk=0
        do n=1,nedge
	     do is=1,npar
	       kk=kk+1
             do j=1,ndof
	         id(j,kk) = ideg(j,n)
	       end do
	     end do
	  end do
c
      if (iprtin.eq.0) then
         nn=0
         do 220 n=1,numnp
         pflag = .false.
c
         do 110 i=1,ndof
         if (id(i,n).ne.0) pflag = .true.
  110    continue
c
         if (pflag) then
            nn = nn + 1
            if (mod(nn,50).eq.1) write(ieco,1100) (i,i=1,ndof)
            write(ieco,2000) n,(id(i,n),i=1,ndof)
         endif
  220    continue
      endif
c
c     establish equation numbers
c
      neq = 0
c
      do 400 n=1,numnp
c
      do 300 i=1,ndof
      if (id(i,n).eq.0) then
         neq = neq + 1
         id(i,n) = neq
      else
         id(i,n) = 1 - id(i,n)
      endif
c
  300 continue
c
  400 continue
c
      return
c
 1000 format(//, 10x, ' e d g e   b o u n d a r y   c o n d i t i o n
     &  c o d e s'///
     & 5x,'   node no.',3x,6(13x,'dof',i1:)//)
 1100 format(//, 10x,' n o d a l   b o u n d a r y   c o n d i t i o n
     &  c o d e s'///
     & 5x,'   node no.',3x,6(13x,'dof',i1:)//)
 2000 format(6x,i10,5x,6(5x,i10))
c
      end

c-----------------------------------------------------------------------
      subroutine bc(id,ndof,numnp,neq,iprtin)
c-----------------------------------------------------------------------
c     program to read, generate and write boundary condition data
c     and establish equation numbers
c-----------------------------------------------------------------------
      dimension id(ndof,*)
c
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      logical pflag
c
      call iclear(id,ndof*numnp)
      call igen(id,ndof)
c
      if (iprtin.eq.0) then
         nn=0
         do 200 n=1,numnp
         pflag = .false.
c
         do 100 i=1,ndof
         if (id(i,n).ne.0) pflag = .true.
  100    continue
c
         if (pflag) then
            nn = nn + 1
            if (mod(nn,50).eq.1) write(ieco,1000) (i,i=1,ndof)
            write(ieco,2000) n,(id(i,n),i=1,ndof)
         endif
  200    continue
      endif
c
c     establish equation numbers
c
      neq = 0
c
      do 400 n=1,numnp
c
      do 300 i=1,ndof
      if (id(i,n).eq.0) then
         neq = neq + 1
         id(i,n) = neq
      else
         id(i,n) = 1 - id(i,n)
      endif
c
  300 continue
c
  400 continue
c
      return
c
 1000 format(///,' n o d a l   b o u n d a r y   c o n d i t i o n  c o
     & d e s'///
     & 5x,' node no.',3x,6(6x,'dof',i1:)//)
 2000 format(6x,i10,5x,6(5x,i10))
c
      end

c-----------------------------------------------------------------------
      block data
c-----------------------------------------------------------------------
c     program to define output labels and numerical constants
c        labeld(3)  = displacement, velocity and acceleration labels
c        label1(16) = output labels for element-type 1
c        label2(3)  = output labels for element-type 2
c
c     note: add label arrays for any additional elements
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /labels/ labeld(3),label1(16),label2(3)

      data   zero,pt1667,pt25,pt5
     &      /0.0d0,0.1666666666666667d0,0.25d0,0.5d0/,
     &       one,two,three,four,five,six
     &      /1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0/
c
!      data labeld/'disp','vel ','acc '/
c
!      data label1/'s 11','s 22','s 12','s 33','ps 1','ps 2',
!     &            'tau ','sang','e 11','e 22','g 12','e 33',
!     &            'pe 1','pe 2','gam ','eang'/
c
!      data label2/'strs','forc','strn'/
c
      end

c-----------------------------------------------------------------------
      subroutine btdb(elstif,b,db,nee,nrowb,nstr)
c-----------------------------------------------------------------------
c     program to multiply b(transpose) * db taking account of symmetry
c     and accumulate into element stiffness matrix
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension elstif(nee,*),b(nrowb,*),db(nrowb,*)
c
      do j=1,nee
         do i=1,j
            elstif(i,j) = elstif(i,j) + coldot(b(1,i),db(1,j),nstr)
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine btod(id,d,brhs,ndof,numnp)
c-----------------------------------------------------------------------
c     program to perform transfer from r.h.s. to displacement array
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension id(ndof,*),d(ndof,*),brhs(*)
c
      do i=1,ndof
         do j=1,numnp
            k = id(i,j)
            if (k.gt.0) d(i,j) = brhs(k)
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine clear(a,m)
c-----------------------------------------------------------------------
c     program to clear a floating-point array
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      do i=1,m
         a(i) = zero
      end do
c
      return
      end

c-----------------------------------------------------------------------
      function coldot(a,b,n)
c-----------------------------------------------------------------------
c     program to compute the dot product of vectors stored column-wise
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(*),b(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      coldot = zero
c
      do i=1,n
         coldot = coldot + a(i)*b(i)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine colht(idiag,lm,ned,nen,numel,neq)
c-----------------------------------------------------------------------
c     program to compute column heights in global left-hand-side matrix
c-----------------------------------------------------------------------
      dimension idiag(*),lm(ned,nen,*)
c
      do 500 k=1,numel
      min = neq
c
      do 200 j=1,nen
c
      do 100 i=1,ned
      num = lm(i,j,k)
      if (num.gt.0) min = min0(min,num)
  100 continue
c
  200 continue
c
      do 400 j=1,nen
c
      do 300 i=1,ned
      num = lm(i,j,k)
      if (num.gt.0) then
         m = num - min
         if (m.gt.idiag(num)) idiag(num) = m
      endif
c
  300 continue

  400 continue
c
  500 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine coord(x,nsd,numnp,iprtin)
c-----------------------------------------------------------------------
c     program to read, generate and write coordinate data
c        x(nsd,numnp) = coordinate array
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension x(nsd,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /genflc/ temp(6,20),n,numgp,ninc(3),inc(3)

      real(8)  :: beta ! parametro para malhas trapezoidais
c
      read(iin,3000) n,numgp,(temp(i,1),i=1,nsd), beta
      backspace(iin)
c
      call genfl(x,nsd)
c
      ! Malhas trapezoidais
      if (beta > 0.d0) then
         do j=2,ninc(2),2
            do i=1,ninc(1)+1,2
               ! baixar
               no = 1 + (j-1)*inc(2) + (i-1)*inc(1)
               noAnty = no-inc(2)
               noProxy = no+inc(2)
               do isd=1,nsd
              x(isd,no)=beta*x(isd,noProxy)+(1.d00-beta)*x(isd,noAnty)
               end do
            end do
            do i=2,ninc(1)+1,2
               ! subir
               no = 1 + (j-1)*inc(2) + (i-1)*inc(1)
               noAnty = no-inc(2)
               noProxy = no+inc(2)
               do isd=1,nsd
              x(isd,no)=beta*x(isd,noAnty)+(1.d00-beta)*x(isd,noProxy)
               end do
            end do
         end do
      end if
c
c
      if (iprtin.eq.1) return
c
      do 100 n=1,numnp
      if (mod(n,50).eq.1) write(ieco,1000) (i,i=1,nsd)
      write(ieco,2000) n,(x(i,n),i=1,nsd)
  100 continue
c
      return
c
 1000 format(///,' n o d a l   c o o r d i n a t e   d a t a '///5x,
     &' node no.',3(13x,' x',i1,' ',:)//)
 2000 format(6x,i10,10x,3(1pe15.8,2x))
 3000 format(2i10,6f10.0)
      end

c-----------------------------------------------------------------------
      subroutine dctnry(name,ndim1,ndim2,ndim3,mpoint,ipr,mlast,ilast)
c-----------------------------------------------------------------------
c     program to store pointer information in dictionary
c-----------------------------------------------------------------------
      character*4 name(2)
      character*4 ia
      common na(1)
      common /dictn/ ia(10000000)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      mlast = mlast - 5
      ia(ilast+1) = name(1)
      ia(ilast+2) = name(2)
      na(mlast+1) = mpoint
      na(mlast+2) = ndim1
      na(mlast+3) = ndim2
      na(mlast+4) = ndim3
      na(mlast+5) = ipr
      ilast = ilast + 2
c
      return
      end

c-----------------------------------------------------------------------
      subroutine diag(idiag,neq,n)
c-----------------------------------------------------------------------
c     program to compute diagonal addresses of left-hand-side matrix
c-----------------------------------------------------------------------
      dimension idiag(*)
c
      n = 1
      idiag(1) = 1
      if (neq.eq.1) return
c
      do 100 i=2,neq
      idiag(i) = idiag(i) + idiag(i-1) + 1
  100 continue
      n = idiag(neq)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine echo
c-----------------------------------------------------------------------
c     program to echo input data
c-----------------------------------------------------------------------
      character*4 ia(20)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      read(iin,1000) iech
      if (iech.eq.0) return
c
      write(ieco,2000) iech
      backspace iin
c
      do 100 i=1,100000
      read(iin,3000,end=200) ia
      if (mod(i,50).eq.1) write(ieco,4000)
      write(ieco,5000) ia
  100 continue
c
  200 continue
      rewind iin
      read(iin,1000) iech
c
      return
c
 1000 format(16i10)
 2000 format(///,' i n p u t   d a t a   f i l e               ',  //5x,
     &' echo print code . . . . . . . . . . . . . . (ieco ) = ',i10//5x,
     &'    eq. 0, no echo of input data                        ',   /5x,
     &'    eq. 1, echo input data                              ',   ///)
 3000 format(20a4)
 4000 format(' ',8('123456789*'),//)
 5000 format(' ',20a4)
      end

c-----------------------------------------------------------------------
      subroutine elemnt(task,ngrp)
c-----------------------------------------------------------------------
c     program to calculate element task number
c-----------------------------------------------------------------------
      character*8 task,eltask(4)
      dimension ngrp(*)
      common /info  / iexec,iprtin,irank,nsd,numnp,ndof,nlvect,
     &                numeg,nmultp,nedge

      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      character*4 ia
      common na(1)
      common /dictn/ ia(10000000)
      data ntask,    eltask
     &    /    4,'input___',
     &           'form_stb',
     &           'form_lrs',
     &           'pos_proc'/
c
      do 100 i=1,ntask
      if (task.eq.eltask(i)) itask = i
  100 continue
c
      do 200 neg=1,numeg
c
      if (itask.eq.1) then
         mpnpar = mpoint('npar    ',16   ,0,0,1)
         ngrp(neg) = mpnpar
         call elcard(na(mpnpar),neg)
      else
         mpnpar = ngrp(neg)
      endif
c
      ntype  = na(mpnpar)
      call elmlib(ntype,mpnpar,itask,neg)
  200 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine elcard(npar,neg)
c-----------------------------------------------------------------------
c     program to read element group control card
c-----------------------------------------------------------------------
      dimension npar(*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      read(iin,1000) (npar(i),i=1,16)
      write(ieco,2000) neg
c
      return
c
 1000 format(16i10)
 2000 format(//,' e l e m e n t   g r o u p   d a t a         ',  //5x,
     &' element group number  . . . . . . . . . . (neg   ) = ',i10/// )
c
      end

c-----------------------------------------------------------------------
      subroutine addnsl(alhs,clhs,eleffm,idiag,lm,nee,ldiag)
c-----------------------------------------------------------------------
c         program to add element left-hand-side matrix to
c                global left-hand-side matrix
c        ldiag = .true.,  add diagonal element matrix
c        ldiag = .false, then
c        add full nonsymmetric element matrix
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical ldiag
      dimension alhs(*),clhs(*),eleffm(nee,*),idiag(*),lm(*)
c
      if (ldiag) then
c
         do 100 j=1,nee
            k = iabs(lm(j))
            if (k.gt.0) then
               l = idiag(k)
               alhs(l) = alhs(l) + eleffm(j,j)
            endif
  100    continue
c
      else
c
         do 400 j=1,nee
            k = iabs(lm(j))
            if (k.gt.0) then
               do 200 i=1,nee
                  m = iabs(lm(i))
                  if (m.gt.0) then
                     if (k.gt.m) then
                        l = idiag(k) - k + m
                        alhs(l) = alhs(l) + eleffm(i,j)
                     else
                        l = idiag(m) - m + k
                        clhs(l) = clhs(l) + eleffm(i,j)
                     endif
                     if (k.eq.m) then
                        l = idiag(k)
                        alhs(l) = alhs(l) + eleffm(i,j)
                        clhs(l) = alhs(l)
                     endif
                  endif
  200          continue
            endif
  400    continue
c
      endif
c
      return
      end

c-----------------------------------------------------------------------
      subroutine backns(a,c,b,idiag,neq)
c-----------------------------------------------------------------------
c     program to perform forward reduction and back substitution
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(*),c(*),b(*),idiag(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
c     forward reduction
c
      jj = 0
c
      do 100 j=1,neq
      jjlast = jj
      jj     = idiag(j)
      jcolht = jj - jjlast
      if (jcolht.gt.1) then
          b(j) = b(j) - coldot(c(jjlast+1),b(j-jcolht+1),jcolht-1)
      endif
  100 continue
c
c     diagonal scaling
c
      do 200 j=1,neq
      ajj = a(idiag(j))
c
c     warning: diagonal scaling is not performed if ajj equals zero
c
      if (ajj.ne.zero) b(j) = b(j)/ajj
  200 continue
c
c     back substitution
c
      if (neq.eq.1) return
      jjnext = idiag(neq)
c
      do 400 j=neq,2,-1
      jj     = jjnext
      jjnext = idiag(j-1)
      jcolht = jj - jjnext
      if (jcolht.gt.1) then
         bj = b(j)
         istart = j - jcolht + 1
         jtemp  = jjnext - istart + 1
c
         do 300 i=istart,j-1
         b(i) = b(i) - a(jtemp+i)*bj
  300    continue
c
      endif
c
  400 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine factns(a,c,idiag,neq)
c-----------------------------------------------------------------------
c     program to perform crout factorization: a = l * d * u
c        a(i):  coefficient matrix stored in compacted column form;
c               after factorization contains d and u
c        c(i):  non-symmetric lower triangular coefficient matrix stored in
c                compacted row form; after factorization contains l
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(*),c(*),idiag(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      do i=1,neq
        write(98,*) i, idiag(i)
      end do
      jj = 0
c
      do 300 j=1,neq
c
      jjlast = jj
      jj     = idiag(j)
      jcolht = jj - jjlast
c
      if (jcolht.gt.2) then
c
c     for column j and i.le.j-1, replace a(i,j) with d(i,i)*u(i,j)
c
         istart = j - jcolht + 2
         jm1    = j - 1
         ij     = jjlast + 2
         ii     = idiag(istart-1)
c
         do 100 i=istart,jm1
c
         iilast = ii
         ii     = idiag(i)
         icolht = ii - iilast
         length = min0(icolht-1,i - istart + 1)
         if (length.gt.0)  then
            a(ij) = a(ij) - coldot(a(ij-length),c(ii-length),length)
            c(ij) = c(ij) - coldot(c(ij-length),a(ii-length),length)
         endif
         ij = ij + 1
  100    continue
c
      endif
c
      if (jcolht.ge.2) then
c
c     for column j and i.le.j-1, replace a(i,j) with u(i,j);
c     replace a(j,j) with d(j,j).
c
         jtemp = j - jj
c
         do 200 ij=jjlast+1,jj-1
c
         ii = idiag(jtemp + ij)
c
c     warning: the following calculations are skipped
c     if a(ii) equals zero
c
         if (a(ii).ne.zero) then
             c(ij) = c(ij)/a(ii)
             a(jj) = a(jj) - c(ij)*a(ij)
             a(ij) = a(ij)/a(ii)
         endif
  200    continue
c
      endif
c
  300 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine factor(a,idiag,neq)
c-----------------------------------------------------------------------
c     program to perform Crout factorization: a = u(transpose) * d * u
c        a(i): coefficient matrix stored in compacted column form;
c              after factorization contains d and u
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension a(*),idiag(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      jj = 0
c
      do 300 j=1,neq
c
      jjlast = jj
      jj     = idiag(j)
      jcolht = jj - jjlast
c
      if (jcolht.gt.2) then
c
c     for column j and i.le.j-1, replace a(i,j) with d(i,i)*u(i,j)
c
         istart = j - jcolht + 2
         jm1    = j - 1
         ij     = jjlast + 2
         ii     = idiag(istart-1)
c
         do 100 i=istart,jm1
c
         iilast = ii
         ii     = idiag(i)
         icolht = ii - iilast
         jlngth = i - istart + 1
         length = min0(icolht-1,jlngth)
         if (length.gt.0)
     &      a(ij) = a(ij) - coldot(a(ii-length),a(ij-length),length)
         ij = ij + 1
  100    continue
c
      endif
c
      if (jcolht.ge.2) then
c
c     for column j and i.le.j-1, replace a(i,j) with u(i,j);
c     replace a(j,j) with d(j,j).
c
         jtemp = j - jj
c
         do 200 ij=jjlast+1,jj-1
c
         ii = idiag(jtemp + ij)
         if (a(ii).ne.zero) then
            temp  = a(ij)
            a(ij) = temp/a(ii)
            a(jj) = a(jj) - temp*a(ij)
         endif
  200    continue
c
      endif
c
  300 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine formlm (id,ien,lm,ndof,ned,nen,numel)
c-----------------------------------------------------------------------
c     program to form lm array
c-----------------------------------------------------------------------
      dimension id(ndof,*),ien(nen,*),lm(ned,nen,*)
c
      do 300 k=1,numel
c
      do 200 j=1,nen
      node=ien(j,k)
c
      do 100 i=1,ndof
      lm(i,j,k) = id(i,node)
  100 continue
c
  200 continue
c
  300 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine ftod(id,d,f,ndof,numnp,nlvect)
c-----------------------------------------------------------------------
c     program to compute displacement boundary conditions
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension id(ndof,*),d(ndof,*),f(ndof,numnp,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      do 300 i=1,ndof
c
         do 200 j=1,numnp
c
            k = id(i,j)
            if (k.gt.0) go to 200
            val = zero
            do 100 lv=1,nlvect
               val = val + f(i,j,lv)
 100        continue
c
            d(i,j) = val
c
 200     continue
c
 300  continue
      return
      end

c-----------------------------------------------------------------------
      subroutine genelad(lado,nside)
c-----------------------------------------------------------------------
c     program to read and generate element node and material numbers
c         lado(nside,numel) = element node numbers
c         mat(numel) e,    = element material numbers
c         nen            = number of element nodes (le.27)
c         n              = element number
c         ng             = generation parameter
c         nel(i)         = number of elements in direction i
c         incel(i)       = element number increment for direction i
c         inc(i)         = node number increment for direction i
c-----------------------------------------------------------------------
      dimension lado(nside,*),itemp(27)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /genelc/ n,nel(3),incel(3),inc(3)
      common /genelf/ incf(3,6)
c
 100  continue
c
c     le a linha de conectividade exemplo do elemento
c
      read(iin,1000) n,(itemp(i),i=1,nside),ng

      write(*,*) "GENELAD", n, (itemp(i), i=1,nside), ng

      if (n.eq.0) return
c
c     grava a conectividade exemplo do elemento
c
      do no =1,nside
         lado(no,n) = itemp(no)
      end do
c
      if (ng.ne.0) then
c
c     gera o resto da conectividade seguindo o padrao dado abaixo
c
c     *** OLD ***
c     read(iin,1000) (nel(i),incel(i),inc(i),incb(i),i=1,3)
c     *** OLD ***

         read(iin,2000) nel(1),incel(1),
     &        incf(1,1),incf(1,2),incf(1,3),
     &        incf(1,4),incf(1,5),incf(1,6)
         read(iin,2000) nel(2),incel(2),
     &        incf(2,1),incf(2,2),incf(2,3),
     &        incf(2,4),incf(2,5),incf(2,6)
         read(iin,1000) nel(3),incel(3),
     &        incf(3,1),incf(3,2),incf(3,3),
     &        incf(3,4),incf(3,5),incf(3,6)
         call genelad1(lado,nside)
      endif
c
      go to 100
c
c
 1000 format(20i10)
 2000 format(8i10)
      end

c-----------------------------------------------------------------------
      subroutine genelad1(ipar,nodsp)
c-----------------------------------------------------------------------
c     program to generate element node and material numbers
c-----------------------------------------------------------------------
      dimension ipar(nodsp,*)
      common /genelc/ n,nel(3),incel(3),inc(3)
      common /genelf/ incf(3,6)
c
c     set defaults
c
      call geneld
c
c     generation algorithm
c
      write(*,*) "GENELAD1"
      ie = n
      je = n
      ke = n
c
      ii = nel(1)
      jj = nel(2)
      kk = nel(3)
c
      do k=1,kk
c
         do j=1,jj
c
            do i=1,ii
               if (i.ne.ii) then
                  le = ie
                  ie = le + incel(1)
                  do l=1,nodsp
                     ipar(l,ie) = ipar(l,le) + incf(1,l)
                  end do
               endif
            end do
c
            if (j.ne.jj) then
               le = je
               je = le + incel(2)
               do l=1,nodsp
                  ipar(l,je) = ipar(l,le) + incf(2,l)
               end do
               ie = je
            endif
         end do
c
         if (k.ne.kk) then
            le = ke
            ke = le + incel(3)
            do l=1,nodsp
               ipar(l,ke) = ipar(l,le) + incf(3,l)
            end do
            ie = ke
c
c     important for 3d (missing in the old code)
c
            je = ke
         endif
c
      end do
c
      return
      end

c$$$c-----------------------------------------------------------------------
c$$$      subroutine genelad1(ipar,nodsp)
c$$$c-----------------------------------------------------------------------
c$$$c     program to generate element node and material numbers
c$$$c-----------------------------------------------------------------------
c$$$      dimension ipar(nodsp,*)
c$$$      common /genelc/ n,nel(3),incel(3),inc(3)
c$$$      common /genelf/ incb(3)
c$$$c
c$$$c     set defaults
c$$$c
c$$$      call geneld
c$$$c
c$$$c     generation algorithm
c$$$c
c$$$      write(*,*) "NODSP genelad1",nodsp
c$$$      ie = n
c$$$      je = n
c$$$      ke = n
c$$$c
c$$$      ii = nel(1)
c$$$      jj = nel(2)
c$$$      kk = nel(3)
c$$$c
c$$$      do k=1,kk
c$$$c
c$$$         do j=1,jj
c$$$            do i=1,ii
c$$$               if (i.ne.ii) then
c$$$                  le = ie
c$$$                  ie = le + incel(1)
c$$$c     call genelip(ipar(1,ie),ipar(1,le),inc(1),nodsp)
c$$$c     *** new
c$$$                  do l=1,nodsp
c$$$                     ipar(l,ie) = ipar(l,le) + inc(1)
c$$$                  end do
c$$$c     *** end new
c$$$               endif
c$$$            end do
c$$$c
c$$$            if (j.ne.jj) then
c$$$c              le = je
c$$$               le = je + (k-1)*incel(2)
c$$$               je = le + incel(2)
c$$$c     call genelip(ipar(1,je),ipar(1,le),inc(2),nodsp)
c$$$c     *** new
c$$$
c$$$c               do l=1, nodsp
c$$$c                  ipar(l,je) = ipar(l,le) + inc(2)
c$$$c              end do
c$$$               km1 = k-1
c$$$               if(km1.eq.0) then
c$$$                  ipar(1,je) = ipar(1,le) + inc(2)
c$$$                  ipar(2,je) = ipar(2,le) + incb(2)
c$$$                  ipar(3,je) = ipar(3,le) + incb(2)
c$$$                  ipar(4,je) = ipar(4,le) + incb(2)
c$$$                  ipar(5,je) = ipar(5,le) + incb(2)
c$$$                  ipar(6,je) = ipar(6,le) + incb(2)
c$$$               endif
c$$$               if(km1.gt.0) then
c$$$                  ipar(1,je) = ipar(1,le) + inc(2)
c$$$                  ipar(2,je) = ipar(2,le) + (incb(2)-km1*nel(2))
c$$$                  ipar(3,je) = ipar(3,le) + (incb(2)-km1*nel(2))
c$$$                  ipar(4,je) = ipar(4,le) + (incb(2)-km1*nel(2))
c$$$                  ipar(5,je) = ipar(5,le) + incb(2)
c$$$                  ipar(6,je) = ipar(6,le) + (incb(2)-km1*nel(2))
c$$$            endif
c$$$c
c$$$c     obs: o termo -km1*nel(2) retira o num faces que
c$$$c          ja foram contadas
c$$$c
c$$$
c$$$c     *** end new
c$$$               ie = je
c$$$            endif
c$$$         end do
c$$$c
c$$$         if (k.ne.kk) then
c$$$            le = ke
c$$$            ke = le + incel(3)
c$$$c     call genelip(ipar(1,ke),ipar(1,le),inc(3),nodsp)
c$$$c     *** new
c$$$c            do l=1, nodsp
c$$$c               ipar(l,ke) = ipar(l, le) + inc(3)
c$$$c            end do
c$$$            ipar(1,ke) = ipar(1,le) + incb(3)
c$$$            ipar(2,ke) = ipar(2,le) + incb(3)
c$$$            ipar(3,ke) = ipar(3,le) + incb(3)
c$$$            ipar(4,ke) = ipar(4,le) + incb(3)
c$$$            ipar(5,ke) = ipar(5,le) + inc(3)
c$$$            ipar(6,ke) = ipar(6,le) + (incb(3)-2)
c$$$c     *** end new
c$$$            ie = ke
c$$$         endif
c$$$c
c$$$      end do
c$$$c
c$$$      return
c$$$      end

c-----------------------------------------------------------------------
      subroutine genelpar(ipar,ien,lado,idside,
     &     nen,nside,nodsp,numel,npars)
c-----------------------------------------------------------------------
c     call genelpar(ipar,ien,lado,idside,
c                   nen,nside,nodsp,numel,npars)
c-----------------------------------------------------------------------
c     gera numeracao dos parametros dos multiplicadores poe elemento
c         ipar(npars*nside,numel) = element parameter numbers
c         nside          = number of element sides
c         nodsp          = number of element parameters (le.27)
c         n              = element number
c         ng             = generation parameter
c         nel(i)         = number of elements in direction i
c         incel(i)       = element number increment for direction i
c         inc(i)         = node number increment for direction i
c-----------------------------------------------------------------------
      dimension lado(nside,*),ipar(nodsp,*)
      dimension ien(nen,*),idside(nside,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do nl=1,numel
         do l=1,nside
            la = lado(l,nl)
            lada = npars*(la-1)
            do np=1,npars
               nsp = (l-1)*npars + np
               ipar(nsp,nl) = lada + np
c               write(*,*) nl, l, nsp, lada+np
            end do
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine genelip(ien2,ien1,inc,nodsp)
c-----------------------------------------------------------------------
c     program to increment element node numbers
c-----------------------------------------------------------------------
      dimension ien1(*),ien2(*)
c
      do 100 i=1,nodsp
      if (ien1(i).eq.0) then
         ien2(i) = 0
      else
         ien2(i) = ien1(i) + inc
      endif
  100 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine genel(ien,mat,nen)
c-----------------------------------------------------------------------
c     program to read and generate element node and material numbers
c         ien(nen,numel) = element node numbers
c         mat(numel)     = element material numbers
c         nen            = number of element nodes (le.27)
c         n              = element number
c         ng             = generation parameter
c         nel(i)         = number of elements in direction i
c         incel(i)       = element number increment for direction i
c         inc(i)         = node number increment for direction i
c-----------------------------------------------------------------------
      dimension ien(nen,*),mat(*),itemp(27)
      common /genelc/ n,nel(3),incel(3),inc(3)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
 100  continue
      read(iin,1000) n,m,(itemp(i),i=1,nen),ng
c
      write(*,*) "subroutine genel"
      write(*,*) "n,m,ng",n,m,ng
      write(*,*) (itemp(i),i=1,nen)
c
      if (n.eq.0) return
      call imove(ien(1,n),itemp,nen)
      mat(n)=m
c
c     generate data
c
      if (ng.ne.0) then
         read(iin,1000) (nel(i),incel(i),inc(i),i=1,3)
         call genel1(ien,mat,nen)
      endif
      go to 100
c
 1000 format(20i10)
c
      end

c-----------------------------------------------------------------------
      subroutine genel1(ien,mat,nen)
c-----------------------------------------------------------------------
c     program to generate element node and material numbers
c-----------------------------------------------------------------------
      dimension ien(nen,*),mat(*)
      common /genelc/ n,nel(3),incel(3),inc(3)
c
c     set defaults
c
      call geneld
c
c     generation algorithm
c
      ie = n
      je = n
      ke = n
c
      ii = nel(1)
      jj = nel(2)
      kk = nel(3)
c
c
      do k=1,kk
         do j=1,jj
            do i=1,ii
c
               if (i.ne.ii) then
                  le = ie
                  ie = le + incel(1)
                  call geneli(ien(1,ie),ien(1,le),inc(1),nen)
                  mat(ie) = mat(le)
               endif
            end do
c
            if (j.ne.jj) then
c     Tentativa antiga de colocar pra 3D....
c               le = je + (k-1)*incel(2)
               le = je
               je = le + incel(2)
               call geneli(ien(1,je),ien(1,le),inc(2),nen)
               mat(je) = mat(le)
               ie = je
            endif
         end do
c
         if (k.ne.kk) then
            le = ke
            ke = le + incel(3)
            call geneli(ien(1,ke),ien(1,le),inc(3),nen)
            mat(ke) = mat(le)
            ie = ke
c
c     Essa linha abaixo nao existia. Para funcionar para malhas 3D
c     foi preciso adiciona-la aqui.
c
            je = ke
         endif
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine geneld
c-----------------------------------------------------------------------
c     program to set defaults for element node
c      and material number generation
c-----------------------------------------------------------------------
      common /genelc/ n,nel(3),incel(3),inc(3)
c
      if (nel(1).eq.0) nel(1) = 1
      if (nel(2).eq.0) nel(2) = 1
      if (nel(3).eq.0) nel(3) = 1
c
      if (incel(1).eq.0) incel(1) = 1
      if (incel(2).eq.0) incel(2) = nel(1)
      if (incel(3).eq.0) incel(3) = nel(1)*nel(2)
c
      if (inc(1).eq.0) inc(1) = 1
      if (inc(2).eq.0) inc(2) = (1+nel(1))*inc(1)
      if (inc(3).eq.0) inc(3) = (1+nel(2))*inc(2)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine geneli(ien2,ien1,inc,nen)
c-----------------------------------------------------------------------
c     program to increment element node numbers
c-----------------------------------------------------------------------
      dimension ien1(*),ien2(*)
c
      do i=1,nen
         if (ien1(i).eq.0) then
            ien2(i) = 0
         else
            ien2(i) = ien1(i) + inc
         endif
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine genfl(a,nra)
c-----------------------------------------------------------------------
c     program to read and generate floating-point nodal data
c         a       = input array
c         nra     = number of rows in a (le.6)
c         n       = node number
c         numgp   = number of generation points
c         ninc(i) = number of increments for direction i
c         inc(i)  = increment for direction i
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      dimension a(nra,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /genflc/ temp(6,20),n,numgp,ninc(3),inc(3)
c
 100  continue
      read(iin,1000) n,numgp,(temp(i,1),i=1,nra)
c
      if (n.eq.0) return
c
      call move(a(1,n),temp,nra)
      if (numgp.ne.0) then
         do j=2,numgp
            read(iin,1000) m,mgen,(temp(i,j),i=1,nra)
            if (mgen.ne.0) call move(temp(1,j),a(1,m),nra)
         end do
         read(iin,2000) (ninc(i),inc(i),i=1,3)
         call genfl1(a,nra)
      endif
      go to 100
c
 1000 format(2i10,6f10.0)
 2000 format(16i10)
c
      end

c-----------------------------------------------------------------------
      subroutine genfl1(a,nra)
c-----------------------------------------------------------------------
c     program to generate floating-point nodal data
c        via isoparametric interpolation
c         iopt = 1, generation along a line
c              = 2, generation over a surface
c              = 3, generation within a volume
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      dimension a(nra,*),sh(20)
      common /genflc/ temp(6,20),n,numgp,ninc(3),inc(3)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      iopt = 3
      if (ninc(3).eq.0) iopt = 2
      if (ninc(2).eq.0) iopt = 1
c
      dr = zero
      ds = zero
      dt = zero
c
      if (ninc(1).ne.0) dr = two/ninc(1)
      if (ninc(2).ne.0) ds = two/ninc(2)
      if (ninc(3).ne.0) dt = two/ninc(3)
c
      ii = ninc(1)+1
      jj = ninc(2)+1
      kk = ninc(3)+1
c
      ni = n
      nj = n
      nk = n
c
      t = -one
      do k=1,kk
         s = -one
         do j=1,jj
            r = -one
            do i=1,ii
               call gensh(r,s,t,sh,numgp,iopt)
               call multab(temp,sh,a(1,ni),6,20,nra,numgp,nra,1,1)
               ni = ni + inc(1)
               r = r + dr
            end do
            nj = nj + inc(2)
            ni = nj
            s = s + ds
         end do
         nk = nk + inc(3)
         ni = nk
         t = t + dt
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine gensh(r,s,t,sh,numgp,iopt)
c-----------------------------------------------------------------------
c     program to call shape function routines
c        for isoparametric generation
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension sh(*)
c
      go to (100,200,300),iopt
c
  100 call gensh1(r,sh,numgp)
      return
c
  200 call gensh2(r,s,sh,numgp)
      return
c
  300 call gensh3(r,s,t,sh,numgp)
      return
c
      end

c-----------------------------------------------------------------------
      subroutine gensh1(r,sh,n)
c-----------------------------------------------------------------------
c     program to compute 1d shape functions
c        for isoparametric generation
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension sh(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      sh(2) = pt5*r
      sh(1) = pt5 - sh(2)
      sh(2) = pt5 + sh(2)
      if (n.eq.3) then
         sh(3) = one - r*r
         sh(1) = sh(1) - pt5*sh(3)
         sh(2) = sh(2) - pt5*sh(3)
      endif
c
      return
      end

c-----------------------------------------------------------------------
      subroutine gensh2(r,s,sh,n)
c-----------------------------------------------------------------------
c     program to compute 2d shape functions
c        for isoparametric generation
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension sh(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      r2 = pt5*r
      r1 = pt5 - r2
      r2 = pt5 + r2
      s2 = pt5*s
      s1 = pt5 - s2
      s2 = pt5 + s2
      sh(1) = r1*s1
      sh(2) = r2*s1
      sh(3) = r2*s2
      sh(4) = r1*s2
      if (n.eq.4) return
c
      r3 = one - r*r
      s3 = one - s*s
      sh(5) = r3*s1
      sh(6) = s3*r2
      sh(7) = r3*s2
      sh(8) = s3*r1
      sh(1) = sh(1) - pt5*(sh(5) + sh(8))
      sh(2) = sh(2) - pt5*(sh(6) + sh(5))
      sh(3) = sh(3) - pt5*(sh(7) + sh(6))
      sh(4) = sh(4) - pt5*(sh(8) + sh(7))
c
      return
      end

c-----------------------------------------------------------------------
      subroutine gensh3(r,s,t,sh,n)
c-----------------------------------------------------------------------
c     program to compute 3d shape functions
c        for isoparametric generation
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension sh(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      r2 = pt5*r
      r1 = pt5 - r2
      r2 = pt5 + r2
      s2 = pt5*s
      s1 = pt5 - s2
      s2 = pt5 + s2
      t2 = pt5*t
      t1 = pt5 - t2
      t2 = pt5 + t2
c
      rs1 = r1*s1
      rs2 = r2*s1
      rs3 = r2*s2
      rs4 = r1*s2
      sh(1) = rs1*t1
      sh(2) = rs2*t1
      sh(3) = rs3*t1
      sh(4) = rs4*t1
      sh(5) = rs1*t2
      sh(6) = rs2*t2
      sh(7) = rs3*t2
      sh(8) = rs4*t2
      if (n.eq.8) return
c
      r3 = one - r*r
      s3 = one - s*s
      t3 = one - t*t
      sh(17) = t3*rs1
      sh(18) = t3*rs2
      sh(19) = t3*rs3
      sh(20) = t3*rs4
      rs1 = r3*s1
      rs2 = s3*r2
      rs3 = r3*s2
      rs4 = s3*r1
      sh( 9) = rs1*t1
      sh(10) = rs2*t1
      sh(11) = rs3*t1
      sh(12) = rs4*t1
      sh(13) = rs1*t2
      sh(14) = rs2*t2
      sh(15) = rs3*t2
      sh(16) = rs4*t2
c
      sh(1) = sh(1) - pt5*(sh( 9) + sh(12) + sh(17))
      sh(2) = sh(2) - pt5*(sh( 9) + sh(10) + sh(18))
      sh(3) = sh(3) - pt5*(sh(10) + sh(11) + sh(19))
      sh(4) = sh(4) - pt5*(sh(11) + sh(12) + sh(20))
      sh(5) = sh(5) - pt5*(sh(13) + sh(16) + sh(17))
      sh(6) = sh(6) - pt5*(sh(13) + sh(14) + sh(18))
      sh(7) = sh(7) - pt5*(sh(14) + sh(15) + sh(19))
      sh(8) = sh(8) - pt5*(sh(15) + sh(16) + sh(20))
c
      return
      end

c-----------------------------------------------------------------------
      subroutine iclear(ia,m)
c-----------------------------------------------------------------------
c     program to clear an integer array
c
      dimension ia(*)
c
      do 100 i=1,m
      ia(i) = 0
  100 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine igen(ia,m)
c-----------------------------------------------------------------------
c     program to read and generate integer nodal data
c        ia = input array
c         m = number of rows in ia
c         n = node number
c        ne = end node in generation sequence
c        ng = generation increment
c-----------------------------------------------------------------------
      dimension ia(m,*),ib(13)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
  100 continue
      read(iin,1000) n,ne,ng,(ib(i),i=1,m)
c
      if (n.eq.0) return
c
      if (ng.eq.0) then
         ne = n
         ng = 1
      else
         ne = ne - mod(ne-n,ng)
      endif
c
      do i=n,ne,ng
         call imove(ia(1,i),ib,m)
      end do
c
      go to 100
c
 1000 format(16i10)
      end

c-----------------------------------------------------------------------
      subroutine imove(ia,ib,n)
c-----------------------------------------------------------------------
c     program to move an integer array
c-----------------------------------------------------------------------
      dimension ia(*),ib(*)
c
      do i=1,n
         ia(i)=ib(i)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine input(f,ndof,numnp,j,nlvect,iprtin)
c-----------------------------------------------------------------------
c     program to read, generate and write nodal input data
c        f(ndof,numnp,nlvect) = prescribed forces/kinematic data (j=0)
c                             = nodal body forces(j=1)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical lzero
      dimension f(ndof,numnp,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      call clear(f,nlvect*numnp*ndof)
c
      do 100 nlv=1,nlvect
      call genfl(f(1,1,nlv),ndof)
      call ztest(f(1,1,nlv),ndof*numnp,lzero)
c
      if (iprtin.eq.0) then
c
         if (lzero) then
            if (j.eq.0) write(ieco,1000) nlv
            if (j.eq.1) write(ieco,2000)
         else
            if (j.eq.0) call printf(f,ndof,numnp,nlv)
c
            if (j.eq.1)
     &      call printd(' n o d a l  b o d y  f o r c e s            ',
     &                  f,ndof,numnp,ieco)
c
         endif
      endif
c
  100 continue
c
      return
 1000 format(/////,' there are no nonzero prescribed forces and ',
     &    'kinematic boundary conditions for load vector number ',i10)
 2000 format(/////,' there are no nonzero nodal body forces')
      end

c-----------------------------------------------------------------------
      subroutine interp(x,y,xx,yy,n)
c-----------------------------------------------------------------------
c     program to perform linear interpolation
c        x(i) = abscissas
c        y(i) = ordinates
c          xx = input abscissa
c          yy = output ordinate
c           n = total number of data points (1.le.i.le.n)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension x(*),y(*)
c
      if (xx.le.x(1)) then
         yy = y(1)
         return
      endif
c
      if (xx.ge.x(n)) then
         yy = y(n)
         return
      endif
c
      do i=1,n
         if (x(i).ge.xx) then
            yy = y(i-1) + (xx - x(i-1))*(y(i) - y(i-1))/(x(i) - x(i-1))
            return
         endif
      end do
c
      end

c-----------------------------------------------------------------------
      subroutine kdbc(eleffm,elresf,dl,nee)
c-----------------------------------------------------------------------
c     program to adjust load vector for prescribed displacement
c     boundary condition
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension eleffm(nee,*),elresf(*),dl(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      do 200 j=1,nee
         val=dl(j)
         if(val.eq.zero) go to 200
         do i=1,nee
            elresf(i)=elresf(i)-eleffm(i,j)*val
         end do
 200  continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine load(id,f,brhs,ndof,numnp,nlvect)
c-----------------------------------------------------------------------
c     program to accumulate nodal forces and transfer into
c        right-hand-side vector
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension id(ndof,*),f(ndof,numnp,*),brhs(*)
c
      do i=1,ndof
         do j=1,numnp
            k = id(i,j)
            if (k.gt.0) then
               do nlv=1,nlvect
                  brhs(k) = brhs(k) + f(i,j,nlv)
               end do
            endif
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine local(ien,x,xl,nen,nrowx,nrowxl)
c-----------------------------------------------------------------------
c     program to localize a global array
c        note: it is assumed nrowxl.le.nrowx
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension ien(*),x(nrowx,*),xl(nrowxl,*)
c
      do j=1,nen
         node = ien(j)
         do i=1,nrowxl
            xl(i,j)= x(i,node)
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      function lout(i,j)
c-----------------------------------------------------------------------
c     program to determine logical switch
c-----------------------------------------------------------------------
      logical lout
c
      lout = .false.
      if (j.eq.0) return
      if (mod(i,j).eq.0) lout = .true.
c
      return
      end

c-----------------------------------------------------------------------
      subroutine matadd(a,b,c,ma,mb,mc,m,n,iopt)
c-----------------------------------------------------------------------
c     program to add rectangular matrices
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
c     remove above card for single-precision operation
c
      dimension a(ma,*),b(mb,*),c(mc,*)
c
      go to (1000,2000,3000),iopt
c
c     iopt = 1, add entire matrices
c
 1000 do j=1,n
         do i=1,m
            c(i,j) = a(i,j) + b(i,j)
         end do
      end do
      return
c
c     iopt = 2, add lower triangular and diagonal elements
c
 2000 do j=1,n
         do i=j,m
            c(i,j) = a(i,j) + b(i,j)
         end do
      end do
      return
c
c     iopt = 3, add upper triangular and diagonal elements
c
 3000 do j=1,n
         do i=1,j
            c(i,j) = a(i,j) + b(i,j)
         end do
      end do
      return
c
      end

c-----------------------------------------------------------------------
      subroutine minmax(x,xmax,xmin,l,m,n)
c-----------------------------------------------------------------------
c
c     program to compute the min and max in the row of a matrix
c
c        x = matrix
c        l = number of rows in x
c        m = number of columns in x
c        n = row number
c
      dimension x(l,*)
c
      xmax = x(n,1)
      xmin = x(n,1)
c
      do 100 i = 2,m
        if (x(n,i).gt.xmax) xmax = x(n,i)
        if (x(n,i).lt.xmin) xmin = x(n,i)
  100 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine move(a,b,n)
c-----------------------------------------------------------------------
c     program to move a floating-point array
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension a(*),b(*)
c
      do i=1,n
         a(i) = b(i)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      function mpoint(name,ndim1,ndim2,ndim3,ipr)
c-----------------------------------------------------------------------
c     program to calculate storage pointer
c-----------------------------------------------------------------------
      character*4 name(2)
      common /bpoint/ mfirst,mlast,ilast,mtot,iprec
c
      mpoint = mfirst
c
      if ( iprec.eq.2 .and. mod(mpoint,2).eq.0 ) mpoint = mpoint + 1
      call dctnry(name,ndim1,ndim2,ndim3,mpoint,ipr,mlast,ilast)
c
      mfirst = mpoint + ndim1*max0(1,ndim2)*max0(1,ndim3)*ipr
c
      if (mfirst.ge.mlast) call serror(name,mfirst-mlast)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine multab(a,b,c,ma,mb,mc,l,m,n,iopt)
c-----------------------------------------------------------------------
c     program to multiply two matrices
c        l = range of dot-product index
c        m = number of active rows in c
c     n = number of active columns in c
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension a(ma,*),b(mb,*),c(mc,*)
c
      go to (1000,2000,3000,4000),iopt
c
c     iopt = 1, c(i,j) = a(i,k)*b(k,j) , (c = a * b)
c
 1000 do i=1,m
         do j=1,n
            c(i,j) = rcdot(a(i,1),b(1,j),ma,l)
         end do
      end do
      return
c                                            t
c     iopt = 2, c(i,j) = a(k,i)*b(k,j) (c = a  * b)
c
 2000 do i=1,m
         do j=1,n
            c(i,j) = coldot(a(1,i),b(1,j),l)
         end do
      end do
      return
c                                                t
c     iopt = 3, c(i,j) = a(i,k)*b(j,k) (c = a * b )
c
 3000 do i=1,m
         do j=1,n
            c(i,j) = rowdot(a(i,1),b(j,1),ma,mb,l)
         end do
      end do
      return
c                                            t    t
c     iopt = 4, c(i,j) = a(k,i)*b(j,k) (c = a  * b )
c
 4000 do i=1,m
         do j=1,n
            c(i,j) = rcdot(b(j,1),a(1,i),mb,l)
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine pivots(a,idiag,neq,nsq,*)
c-----------------------------------------------------------------------
c     program to determine the number of zero and negative terms in
c        array d of factorization a = u(transpose) * d * u
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension a(*),idiag(*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      iz = 0
      in = 0
c
      do n=1,neq
         i = idiag(n)
         if (a(i).eq.0.) iz = iz + 1
         if (a(i).lt.0.) in = in + 1
      end do
c
      write(ieco,1000) nsq,iz,in
c
      return 1
c
 1000 format(' ',
     &' zero and/or negative pivots encountered               ', ///5x,
     &' time sequence number   . . . . . . . . . . (nsq  ) = ',i10//5x,
     &' number of zeros . . . . . . . . . . . . . . . . .  = ',i10//5x,
     &' number of negatives  . . . . . . . . . . . . . . . = ',i10//5x)
c
      end

c-----------------------------------------------------------------------
      subroutine princ(n,s,p)
c-----------------------------------------------------------------------
c     program to compute principal values of symmetric 2nd-rank tensor
c        s = symmetric second-rank tensor stored as a vector
c        n = number of dimensions (2 or 3)
c        p = vector of principal values
c     the components of s must be stored in the following orders
c        2-d problems: s11,s22,s12
c        3-d problems: s11,s22,s33,s12,s23,s31
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension s(*),p(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data rt2/1.41421356237309d0/,pi23/2.09439510239321d0/
c
      if (n.eq.2) then
c
c     2D problem
c
         a = 22.5d0/datan(one)
         x = pt5*(s(1) + s(2))
         y = pt5*(s(1) - s(2))
         r = dsqrt(y*y + s(3)*s(3))
         p(1) = x + r
         p(2) = x - r
         p(3) = r
         p(4) = 45.0d0
         if (y.ne.zero.or.s(3).ne.zero) p(4) = a*atan2(s(3),y)
      endif
c
      if (n.eq.3) then
c
c     3D problem
c
         r = zero
         x = (s(1) + s(2) + s(3))/three
         y = s(1)*(s(2) + s(3)) + s(2)*s(3)
     &       - s(4)*s(4) - s(6)*s(6) - s(5)*s(5)
         z = s(1)*s(2)*s(3) - two*s(4)*s(6)*s(5) - s(1)*s(5)*s(5)
     &       - s(2)*s(6)*s(6) - s(3)*s(4)*s(4)
         t = three*x*x - y
         u = zero
         if (t.ne.zero) then
            u = dsqrt(two*t/three)
            a = (z + (t - x*x)*x)*rt2/u**3
            r = dsqrt(dabs(one - a*a))
            r = datan2(r,a)/three
         endif
         p(1) = x + u*rt2*cos(r)
         p(2) = x + u*rt2*cos(r - pi23)
         p(3) = x + u*rt2*cos(r + pi23)
      endif
c
      return
      end

c-----------------------------------------------------------------------
      subroutine printd(name,dva,ndof,numnp,icode)
c-----------------------------------------------------------------------
c     program to print kinematic data
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      logical lzero
      character *4 name(11)
      dimension dva(ndof,*)
c
      nn = 0
c
      do n=1,numnp
         call ztest(dva(1,n),ndof,lzero)
         if (.not.lzero) then
            nn = nn + 1
            if (mod(nn,50).eq.1)
     &           write(icode,1000) name,(i,i=1,ndof)
            write(icode,2000) n,(dva(i,n),i=1,ndof)
         endif
      end do
c
      return
c
 1000 format(///,11a4//6xx,'node',6(11x,'dof',i1)/)
 2000 format(1x,i10,2x,6(1pe13.6,2x))
      end

c-----------------------------------------------------------------------
      subroutine printf(f,ndof,numnp,nlv)
c-----------------------------------------------------------------------
c     program to print prescribed force and boundary condition data
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      logical lzero
      dimension f(ndof,numnp,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      nn = 0
c
      do n=1,numnp
         call ztest(f(1,n,nlv),ndof,lzero)
         if (.not.lzero) then
            nn = nn + 1
            if (mod(nn,50).eq.1)
     &           write(ieco,1000) nlv,(i,i=1,ndof)
            write(ieco,2000) n,(f(i,n,nlv),i=1,ndof)
         endif
      end do
c
      return
c
 1000 format(///,
     &' p r e s c r i b e d   f o r c e s   a n d   k i n e m a t i c ',
     &'  b o u n d a r y   c o n d i t i o n s'//5x,
     &' load vector number = ',i10///5x,
     &' node no.',6(13x,'dof',i1,:)/)
 2000 format(6x,i10,10x,6(1pe15.8,2x))
      end

c-----------------------------------------------------------------------
      subroutine printp(a,idiag,neq,nsq,*)
c-----------------------------------------------------------------------
c     program to print array d after Crout factorization
c        a = u(transpose) * d * u
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      dimension a(*),idiag(*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do 100 n=1,neq
      if (mod(n,50).eq.1) write(ieco,1000) nsq
      write(ieco,1000)
      i = idiag(n)
      write(ieco,2000) n,a(i)
  100 continue
c
      return 1
c
 1000 format(///,' array d of factorization',/
     &' a = u(transpose) * d * u ',                               //5x,
     &' time sequence number   . . . . . . . . . . . (nsq) = ',i10//5x)
 2000 format(1x,i10,4x,1pe20.8)
      end

c-----------------------------------------------------------------------
      subroutine prntels(mat,ien,nen,numel)
c-----------------------------------------------------------------------
c     program to print data for element with "nen" nodes
c        note: presently the label formats are limited to
c              elements with one to nine nodes
c-----------------------------------------------------------------------
      dimension mat(*),ien(nen,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do 100 n=1,numel
      if (mod(n,50).eq.1) write(ieco,1000) (i,i=1,nen)
      write(ieco,2000) n,mat(n),(ien(i,n),i=1,nen)
  100 continue
c
      return
c
 1000 format(///,
     &' d a t a   f o r    e l e m e n t   s i d e s ',//1x,
     &' element      material',16('      node',i2))
 2000 format(1x,i10,20(2x,i10))
      end

c-----------------------------------------------------------------------
      subroutine prntelp(mat,ien,nen,numel)
c-----------------------------------------------------------------------
c     program to print data for element with "nen" nodes
c        note: presently the label formats are limited to
c              elements with one to nine nodes
c-----------------------------------------------------------------------
      dimension mat(*),ien(nen,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do 100 n=1,numel
      if (mod(n,50).eq.1) then
	  if(nen.le.16) write(ieco,1000) (i,i=1,nen)
	  if(nen.eq.20) write(ieco,1020) (i,i=1,nen)
	  if(nen.eq.24) write(ieco,1024) (i,i=1,nen)
	  if(nen.eq.28) write(ieco,1028) (i,i=1,nen)
	  if(nen.eq.32) write(ieco,1032) (i,i=1,nen)
	end if
      write(ieco,2000) n,mat(n),(ien(i,n),i=1,nen)
  100 continue
c
      return
c
 1000 format(///,
     &' d a t a   f o r   e l e m e n t   p a r a m e t e r s ',//1x,
     &' element      material',16('      node',i2))
 1020 format(///,
     &' d a t a   f o r   e l e m e n t   p a r a m e t e r s ',//1x,
     &' element      material',20('      node',i2))
 1024 format(///,
     &' d a t a   f o r   e l e m e n t   p a r a m e t e r s ',//1x,
     &' element      material',24('      node',i2))
 1028 format(///,
     &' d a t a   f o r   e l e m e n t   p a r a m e t e r s ',//1x,
     &' element      material',28('      node',i2))
 1032 format(///,
     &' d a t a   f o r   e l e m e n t   p a r a m e t e r s ',//1x,
     &' element      material',32('      node',i2))
2000  format(1x,i10,64(2x,i10))
      end

c-----------------------------------------------------------------------
      subroutine prntel(mat,ien,nen,numel)
c-----------------------------------------------------------------------
c     program to print data for element with "nen" nodes
c
c        note: presently the label formats are limited to
c              elements with one to nine nodes
c-----------------------------------------------------------------------
      dimension mat(*),ien(nen,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do 100 n=1,numel
      if (mod(n,50).eq.1) write(ieco,1000) (i,i=1,nen)
      write(ieco,2000) n,mat(n),(ien(i,n),i=1,nen)
  100 continue
c
      return
c
 1000 format(///,
     &' d a t a   f o r   e l e m e n t   n o d e s ',//1x,
     &' element      material',16('      node',i2))
 2000 format(1x,i10,20(2x,i10))
      end

c-----------------------------------------------------------------------
      subroutine prtdc
c-----------------------------------------------------------------------
c     program to print memory-pointer dictionary
c-----------------------------------------------------------------------
      common /bpoint/ mfirst,mlast,ilast,mtot,iprec
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      character *4 ia
      common na(1)
      common /dictn/ ia(10000000)
c
      n = (mtot-mlast)/5
      j = mtot + 1
c
      k = 1
      do 100 i=1,n
      if (mod(i,50).eq.1) write(ipmx,1000)
      j = j - 5
      call prtdc1(i,ia(k),na(j),na(j+1),na(j+2),na(j+3),na(j+4))
      k = k + 2
 100  continue
c
      return
c
 1000 format(///,
     &' d y n a m i c   s t o r a g e    a l l o c a t i o n',
     &'   i n f o r m a t i o n '//
     &  12x,'array no.',5x,'array',8x,'address',6x,'dim1',6x,'dim2',
     &  6x, 'dim3',6x,'prec.'/)
c
      end

c-----------------------------------------------------------------------
      subroutine prtdc1(i,iname,iadd,ndim1,ndim2,ndim3,ipr)
c-----------------------------------------------------------------------
c     program to print memory-pointer information for an array
c-----------------------------------------------------------------------
      character *4 iname(2)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      if (i.eq.1) neg = 1
      if (iname(1).eq.'npar') then
        write (ipmx,1000) neg
        neg = neg + 1
      endif
      write(ipmx,2000) i,iname,iadd,ndim1,ndim2,ndim3,ipr
c
      return
c
 1000 format(/14x,'*****',7x,'begin element group number',i10/' ')
 2000 format(14x,i10,7x,2a4,1x,6i10)
      end

c-----------------------------------------------------------------------
      function rcdot(a,b,ma,n)
c-----------------------------------------------------------------------
c     program to compute the dot product of a vector stored row-wise
c     with a vector stored column-wise
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension a(ma,*),b(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      rcdot = zero
c
      do i=1,n
         rcdot = rcdot + a(1,i)*b(i)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      function rowdot(a,b,ma,mb,n)
c-----------------------------------------------------------------------
c     program to compute the dot product of vectors stored row-wise
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension a(ma,*),b(mb,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      rowdot = zero
c
      do i=1,n
         rowdot = rowdot + a(1,i)*b(1,i)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine cross(a,b,c)
c-----------------------------------------------------------------------
c     program to compute the cross product of vectors (size 3)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(3),b(3),c(3)
c
      c(1) = a(2)*b(3) - a(3)*b(2)
      c(2) = a(3)*b(1) - a(1)*b(3)
      c(3) = a(1)*b(2) - a(2)*b(1)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine nrm3(vec)
c-----------------------------------------------------------------------
c     program to normalize a vector
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension vec(3)
c
      dnorm = dsqrt(vec(1)*vec(1) + vec(2)*vec(2) + vec(3)*vec(3))
      vec(1) = vec(1)/dnorm
      vec(2) = vec(2)/dnorm
      vec(3) = vec(3)/dnorm
c
      return
      end

c-----------------------------------------------------------------------
      function dot3(a,b)
c-----------------------------------------------------------------------
c     program to compute the cross product of vectors (size 3)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(3),b(3)
      dot3 = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
      return
      end

c-----------------------------------------------------------------------
      subroutine vec3sub(a,b,c)
c-----------------------------------------------------------------------
c     program to compute c_i = a_i - b_i
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension a(3),b(3),c(3)
      c(1) = a(1) - b(1)
      c(2) = a(2) - b(2)
      c(3) = a(3) - b(3)
      return
      end

c-----------------------------------------------------------------------
      subroutine centroid(xls,nesd,nen,c)
c-----------------------------------------------------------------------
c     program to compute c_i = a_i - b_i
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension xls(nesd,*), c(nesd)
c
      do j=1,nesd
        c(j) = 0.d0
      end do
c
      do i=1,nen
        do j=1,nesd
         c(j) = c(j) + xls(j,i)
       end do
      end do
c
      do j=1,nesd
        c(j) = c(j)/nen
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine serror(name,i)
c-----------------------------------------------------------------------
c     program to print error message if available storage is exceeded
c-----------------------------------------------------------------------
      character*4 name(2)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      call prtdc
      write(*,1000) i,name
      write(*,*) "pause"
c      pause
      stop
c
 1000 format(1x,5('*'),'storage exceeded by ',i10,
     &' words in attempting to store array ',2a4)
      end

c-----------------------------------------------------------------------
      subroutine setupd(c,dmat,const,nstr,nrowb)
c-----------------------------------------------------------------------
c     program to calculate the d matrix
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension c(nrowb,*),dmat(nrowb,*)
c
      do j=1,nstr
         do i=1,j
            dmat(i,j) = const*c(i,j)
            dmat(j,i) = dmat(i,j)
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shgq4(xl,det,shl,shg,nint,nel,neg,quad)
c-----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions and
c        jacobian determinants for a 4-node quadrilateral element
c        xl(j,i)    = global coordinates
c        det(l)     = jacobian determinant
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c        shg(1,i,l) = x-derivative of shape function
c        shg(2,i,l) = y-derivative of shape function
c        shg(3,i,l) = shl(3,i,l)
c        xs(i,j)    = jacobian matrix
c                 i = local node number or global coordinate number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical quad
      dimension xl(2,*),det(*),shl(3,4,*),shg(3,4,*),xs(2,2)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      call move(shg,shl,12*nint)
c
      do 700 l=1,nint
c
      if (.not.quad) then
         do 100 i=1,3
         shg(i,3,l) = shl(i,3,l) + shl(i,4,l)
         shg(i,4,l) = zero
  100    continue
      endif
c
      do 300 j=1,2
      do 200 i=1,2
      xs(i,j) = rowdot(shg(i,1,l),xl(j,1),3,2,4)
  200 continue
  300 continue
c
      det(l) = xs(1,1)*xs(2,2)-xs(1,2)*xs(2,1)
      if (det(l).le.zero) then
         write(ieco,1000) nel,neg
         stop
      endif
c
      do 500 j=1,2
      do 400 i=1,2
      xs(i,j) = xs(i,j)/det(l)
  400 continue
  500 continue
c
      do 600 i=1,4
        temp = xs(2,2)*shg(1,i,l) - xs(1,2)*shg(2,i,l)
        shg(2,i,l) = - xs(2,1)*shg(1,i,l) + xs(1,1)*shg(2,i,l)
        shg(1,i,l) = temp
  600 continue
c
  700 continue
c
      return
c
 1000 format(///,'non-positive determinant in element number  ',i10,
     &          ' in element group  ',i10)
      end

c-----------------------------------------------------------------------
      subroutine shlq4(shl,w,nint)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a four-node quadrilateral element
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension shl(3,4,*),w(*),ra(4),sa(4)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data ra/-0.5d0,0.5d0,0.5d0,-0.5d0/,sa/-0.5d0,-0.5d0,0.5d0,0.5d0/
c
      g = zero
      w(1) = four
      if (nint.eq.4) then
         g = two/dsqrt(three)
         w(1) = one
         w(2) = one
         w(3) = one
         w(4) = one
      endif
c
      do 200 l=1,nint
      r = g*ra(l)
      s = g*sa(l)
c
      do 100 i=1,4
      tempr = pt5 + ra(i)*r
      temps = pt5 + sa(i)*s
      shl(1,i,l) = ra(i)*temps
      shl(2,i,l) = tempr*sa(i)
      shl(3,i,l) = tempr*temps
  100 continue
c
  200 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine smult(a,b,c,mb,mc,m,n,iopt)
c-----------------------------------------------------------------------
c     program to perform scalar multiplication of a matrix
c        c(i,j) = a*b(i,j)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension b(mb,*),c(mc,*)
c
      go to (1000,2000,3000),iopt
c
c     iopt = 1, multiply entire matrix
c
 1000 do 1200 j=1,n
c
      do 1100 i=1,m
      c(i,j) = a*b(i,j)
 1100 continue
c
 1200 continue
      return
c
c     iopt = 2, multiply lower triangular and diagonal elements
c
 2000 do 2200 j=1,n
c
      do 2100 i=j,m
      c(i,j) = a*b(i,j)
 2100 continue
c
 2200 continue
      return
c
c     iopt = 3, multiply upper triangular and diagonal elements
c
 3000 do 3200 j=1,n
c
      do 3100 i=1,j
      c(i,j) = a*b(i,j)
 3100 continue
c
 3200 continue
      return
c
      end

c-----------------------------------------------------------------------
      subroutine timing(temp)
c-----------------------------------------------------------------------
c     program to determine elapsed cpu time
c     **** this is a system-dependent routine ****
c         note: can only access clock time on vax/vms
!       call time(itemp)
       temp=itemp
c
      return
      end

c-----------------------------------------------------------------------
      subroutine timlog
c-----------------------------------------------------------------------
c     program to print log of execution times
c-----------------------------------------------------------------------
      common /etimec/ etime(6)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /titlec/ title(20)
c
      subtot = 0.0
      do 100 i=3,6
      subtot = subtot + etime(i)
  100 continue
c
      write(ieco,1000) title,(etime(i),i=1,6),subtot
c
      return
c
 1000 format(///,20a4///5x,
     &' e x e c u t i o n   t i m i n g   i n f o r m a t i o n'  ///5x,
     &' i n p u t   p h a s e                          = ',1pe10.3///5x,
     &' s o l u t i o n   p h a s e                    = ',1pe10.3///5x,
     &'     formation of left and right-hand-sides     = ',1pe10.3 //5x,
     &'     factorization                              = ',1pe10.3 //5x,
     &'     forward reduction/back substitution        = ',1pe10.3 //5x,
     &'     post-processing                            = ',1pe10.3  /5x,
     &51x,'_________',//5x
     &'     subtotal                                   = ',1pe10.3     )
c
      end

c-----------------------------------------------------------------------
      subroutine ztest(a,n,lzero)
c-----------------------------------------------------------------------
c     program to determine if an array contains only zero entries
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension a(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      logical lzero
c
      lzero = .true.
c
      do 100 i=1,n
      if (a(i).ne.zero) then
         lzero = .false.
         return
      endif
  100 continue
c
      end

c-----------------------------------------------------------------------
      subroutine shap2m(s,xl,det,sh,nen,ien,nesd)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension xl(nesd,*),sh(2,*),ien(*)
c
c     shape function
c
      sh(2,1)=(1.d00-s)/2.d00
      sh(2,2)=(1.d00+s)/2.d00
c
c
      sh(1,1)=-.5d00
      sh(1,2)=.5d00
c
c     3 node correction
c
      if(nen.eq.3) then
      corr=1.d00-s*s
      corrh=corr/2
      sh(2,1)=sh(2,1)-corrh
      sh(2,2)=sh(2,2)-corrh
      sh(2,3)=corr
c
      corr=-2.d00*s
      corrh=-s
      sh(1,1)=sh(1,1)-corrh
      sh(1,2)=sh(1,2)-corrh
      sh(1,3)=corr
c
      end if
c
      det=0.d00
      do 100 l=1,nen
      det=det+xl(1,l)*sh(1,l)
100   continue
c
c     global derivatives
c
      do 200 l=1,nen
      sh(1,l)=sh(1,l)/det
200   continue
      return
      end

c-----------------------------------------------------------------------
       subroutine shap20(s,t,x,det,sh,sh2,nen,inc,ien,quad)
c-----------------------------------------------------------------------
c     program to compute shape functions for quadrilateral
c        s,t        = natural coordinates
c        sh(nsd,i)  = first derivatives of shape functions
c        sh(3,i)    = shape functions
c        sh2(3,i)   = second derivatives of shape functions
c        xs(nsd,nsd)= jacobian matrix
c        det        = jacobian determinant
c        x(nsd,nen) = global coordinates
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      logical quad
      dimension sa(4),ta(4),sh(3,*),sh2(3,*),x(2,*),xs(2,2)
      dimension ien(1)
      data sa/-0.5d0,0.5d0,0.5d0,-0.5d0/,ta/-0.5d0,-0.5d0,0.5d0,0.5d0/
c
      do 10 i=1,4
      sh(3,i)=(0.5d0+sa(i)*s)*(0.5d0+ta(i)*t)
      sh(1,i)=sa(i)*(0.5d0+ta(i)*t)
      sh(2,i)=ta(i)*(0.5d0+sa(i)*s)
10    continue
c
      if(quad) goto 30
      do 20 i=1,3
      sh(i,3)=sh(i,3)+sh(i,4)
      sh(i,4)=0.d0
20    continue
c
30    if(nen.eq.8.or.nen.eq.9) call shap21(s,t,sh,nen,ien)
c
      do 40 i=1,2
      do 40 j=1,2
      xs(i,j)=0.d0
      do 40 k=1,nen
      xs(i,j)=xs(i,j)+x(i,k)*sh(j,k)
40    continue
      det=xs(1,1)*xs(2,2)-xs(1,2)*xs(2,1)
c
      do 50 i=1,2
      do 50 j=1,2
      xs(i,j)=xs(i,j)/det
50    continue

c      call shap22(s,t,xs,x,sh,sh2,ien,nen)

      do 60 i=1,nen
      temp=xs(2,2)*sh(1,i)-xs(2,1)*sh(2,i)
      sh(2,i)=-xs(1,2)*sh(1,i)+xs(1,1)*sh(2,i)
      sh(1,i)=temp
60    continue

      if(inc.eq.0) return
c
c     incompatible modes
c
      if(quad) goto 80
      do 70 i=1,3
      do 70 j=5,6
   70 sh(i,j)=0.d0
      return
   80 sh(1,5)=-s-s
      sh(2,5)=0.d0
      sh(3,5)=1.d0-s*s
      sh(1,6)=0.d0
      sh(2,6)=-t-t
      sh(3,6)=1.d0-t*t
      xs(1,1)=0.25d0*(-x(1,1)+x(1,2)+x(1,3)-x(1,4))
      xs(1,2)=0.25d0*(-x(1,1)-x(1,2)+x(1,3)+x(1,4))
      xs(2,1)=0.25d0*(-x(2,1)+x(2,2)+x(2,3)-x(2,4))
      xs(2,2)=0.25d0*(-x(2,1)-x(2,2)+x(2,3)+x(2,4))
      do 90 i=5,6
      temp=(xs(2,2)*sh(1,i)-xs(2,1)*sh(2,i))/det
      sh(2,i)=(-xs(1,2)*sh(1,i)+xs(1,1)*sh(2,i))/det
   90 sh(1,i)=temp
      return
      end

c-----------------------------------------------------------------------
      subroutine shap21 (s,t,sh,nen,ien)
c-----------------------------------------------------------------------
c   program to compute shape functions and local derivatives
c   for quadrilateral -  shape functions  5  to  9
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension sh(3,*),ien(*)
      data zero/0.d0/
c
      ss=(1.d0-s*s)/2.d0
      tt=(1.d0-t*t)/2.d0
      do 10 i=5,8
      do 10 j=1,3
   10 sh(j,i)=zero
      s19=zero
      s29=zero
      s39=zero
c
      if(nen.ne.9) goto 15
      if(ien(9).eq.0) goto 15
      s19=-2.d0*tt*s
      s29=-2.d0*ss*t
      s39=2.d0*tt*ss
      sh(1,9)=2.d0*s19
      sh(2,9)=2.d0*s29
      sh(3,9)=2.d0*s39
      do 20 i=1,4
      do 20 j=1,3
   20 sh(j,i)=sh(j,i)-sh(j,9)/4.d0
c
   15 if(ien(5).eq.0) goto 30
      sh(1,5)=-s*(1.d0-t)-s19
      sh(2,5)=-ss-s29
      sh(3,5)=ss*(1.d0-t)-s39
c
   30 if(nen.lt.6) goto 60
      if(ien(6).eq.0) goto 40
      sh(1,6)=tt-s19
      sh(2,6)=-t*(1.d0+s)-s29
      sh(3,6)=tt*(1.d0+s)-s39
c
   40 if(nen.lt.7) goto 60
      if(ien(7).eq.0) goto 50
      sh(1,7)=-s*(1.d0+t)-s19
      sh(2,7)=ss-s29
      sh(3,7)=ss*(1.d0+t)-s39
c
   50 if(nen.lt.8) goto 60
      if(ien(8).eq.0) goto 60
      sh(1,8)=-tt-s19
      sh(2,8)=-t*(1.d0-s)-s29
      sh(3,8)=tt*(1.d0-s)-s39
c
   60 k=8
      do 70 i=1,4
      l=i+4
      do 80 j=1,3
   80 sh(j,i)=sh(j,i)-0.5d0*(sh(j,k)+sh(j,l))
   70 k=l
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shap22(s,t,xs,x,sh,sh2,ien,nen)
c-----------------------------------------------------------------------
c   compute second derivatives of shape functions 1 to 9 for
c   quadrilateral
c          eh(1,i)  =  d2(Ni)/de2
c          eh(2,i)  =  d2(Ni)/dn2
c     eh(3,i)  =  d2(Ni)/dedn
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension sh(3,*),sh2(3,*),ien(*),xs(2,2),x(2,*),eh(3,9)
      call clear(eh,27)
c
c     local second derivatives of shape functions
c
      eh(3,1)= 0.25d0
      eh(3,2)=-0.25d0
      eh(3,3)= 0.25d0
      eh(3,4)=-0.25d0
c
      if(nen.le.4) goto 110
c
      if(nen.ne.9) goto 30
      if(ien(9).eq.0) goto 30
      eh(1,9)=-2.d0*(1.d0-t*t)
      eh(2,9)=-2.d0*(1.d0-s*s)
      eh(3,9)=4.d0*s*t
      do 40 i=1,4
      do 40 j=1,3
   40 eh(j,i)=eh(j,i)-eh(j,9)/4.d0
   30 e19=0.5d0*eh(1,9)
      e29=0.5d0*eh(2,9)
      e39=0.5d0*eh(3,9)
c
      if(ien(5).eq.0) goto 50
      eh(1,5)=-1.d0+t-e19
      eh(2,5)=-e29
      eh(3,5)=s-e39
c
   50 if(nen.lt.6) goto 80
      if(ien(6).eq.0) goto 60
      eh(1,6)=-e19
      eh(2,6)=-1.d0-s-e29
      eh(3,6)=-t-e39
c
   60 if(nen.lt.7) goto 80
      if (ien(7).eq.0) goto 70
      eh(1,7)=-1.d0-t-e19
      eh(2,7)=-e29
      eh(3,7)=-s-e39
c
   70 if(nen.lt.8) goto 80
      if(ien(8).eq.0) goto 80
      eh(1,8)=-e19
      eh(2,8)=-1.d0+s-e29
      eh(3,8)=t-e39
c
   80 k=8
      do 90 i=1,4
      l=i+4
      do 100 j=1,3
  100 eh(j,i)=eh(j,i)-0.5d0*(eh(j,k)+eh(j,l))
   90 k=l
c
c     global second derivatives
c
110   call shap23(xs,x,eh,sh,sh2,nen)
      return
      end

c-----------------------------------------------------------------------
      subroutine shap23(xs,x,eh,sh,sh2,nen)
c-----------------------------------------------------------------------
c   transform second derivatives from natural coordinates to
c   global coordinates
c         sh2(1,i)  =  d2(Ni)/dx2
c         sh2(2,i)  =  d2(Ni)/dy2
c         sh2(3,i)  =  d2(Ni)/dxdy
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension xs(2,2),x(2,*),eh(3,*),sh(3,*),sh2(3,*)
      dimension t2(3,3),c1(3,2),t1(3,2),xj(2,2)
c
      call clear(c1,6)
      call clear(t1,6)
      call clear(sh2,3*nen)
c
c     form j inverse of jacobian matrix
c
      xj(1,1)=xs(2,2)
      xj(2,2)=xs(1,1)
      xj(1,2)=-xs(2,1)
      xj(2,1)=-xs(1,2)
c
c     form t2
c
      do 10 i=1,2
      t2(i,3)=2.d0*xj(i,1)*xj(i,2)
      t2(3,i)=xj(1,i)*xj(2,i)
      do 10 j=1,2
      t2(i,j)=xj(i,j)**2
10    continue
      t2(3,3)=xj(1,1)*xj(2,2)+xj(1,2)*xj(2,1)
c
c     form c1
c
      do 20 n=1,3
      do 20 i=1,2
      do 20 j=1,nen
      c1(n,i)=c1(n,i)+eh(n,j)*x(i,j)
20    continue
c
c     form t1
c
      do 30 i=1,3
      do 30 j=1,2
      do 30 k=1,3
      t1(i,j)=t1(i,j)-t2(i,k)*(c1(k,1)*xj(1,j)+c1(k,2)*xj(2,j))
30    continue
c
c     transformation from natural coor. to global coor.
c
      do 50 n=1,nen
      do 50 l=1,3
         do 60 i=1,2
         sh2(l,n)=sh2(l,n)+t1(l,i)*sh(i,n)
60       continue
         do 70 i=1,3
         sh2(l,n)=sh2(l,n)+t2(l,i)*eh(i,n)
70       continue
50    continue
c
      return
      end

c***********************************************************************
c          shape functions
c***********************************************************************

c-----------------------------------------------------------------------
      subroutine oneshl(shl,w,nint,nen)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a one-dimensional element
c                 r = local element coordinate ("xi")
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration-point number
c              nint = number of integration points, eq. 1,2,3,4,6,7,8
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension shl(3,nen,*),w(*),ra(10),xa(10)
      data   zero,pt1667,pt25,pt5
     &      /0.0d0,0.1666666666666667d0,0.25d0,0.5d0/,
     &       one,two,three,four,five,six
     &      /1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0/

      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
c
      if (nint.eq.1) then
         w(1)  = two
         ra(1) = zero
      endif
c
      if (nint.eq.2) then
         w(1) = one
         w(2) = one
         ra(1)=-.577350269189626
         ra(2)= .577350269189625
      endif
c
      if (nint.eq.3) then
         w(1) = five9
         w(2) = eight9
         w(3) = five9
         ra(1)=-.774596669241483
         ra(2)= zero
         ra(3)= .774596669241483
      endif
c
      if (nint.eq.4) then
         w(1) = .347854845137454
         w(2) = .652145154862546
         w(3) = .652145154862546
         w(4) = .347854845137454
         ra(1)=-.861136311594053
         ra(2)=-.339981043584856
         ra(3)= .339981043584856
         ra(4)= .861136311594053
      endif
c
       if(nint.eq.5) then
        w(1) = .236926885056189
        w(2) = .478628670499366
        w(3) = .568888888888888
        w(4) = .478628670499366
        w(5) = .236926885056189
        ra(1)=-.906179845938664
        ra(2)=-.538469310105683
        ra(3)= zero
        ra(4)= .538469310105683
        ra(5)= .906179845938664
       endif
c
       if(nint.eq.6) then
         w(1) = .171324492397170
         w(2) = .360761573048139
         w(3) = .467913934572691
         w(4) = .467913934572691
         w(5) = .360761573048139
         w(6) = .171324492397170
c
         ra(1)=-.932469514203152
         ra(2)=-.661209386466265
         ra(3)=-.238619186083197
         ra(4)= .238619186083197
         ra(5)= .661209386466365
         ra(6)= .932469514203152
        endif
c
       if(nint.eq.7) then
         w(1) = .129484966168870
         w(2) = .279705391489277
         w(3) = .381830050505119
         w(4) = .417959183673469
         w(5) = .381830050505119
         w(6) = .279705391489277
         w(7) = .129484966168870
c
         ra(1)=-.949107912342759
         ra(2)=-.741531185599394
         ra(3)=-.405845151377397
         ra(4)= zero
         ra(5)= .405845151377397
         ra(6)= .741531185599394
         ra(7)= .949107912342759
        endif
c
       if(nint.eq.8) then
         w(1) = .101228536290376
         w(2) = .222381034453374
         w(3) = .313706645877887
         w(4) = .362683783378362
         w(5) = .362683783378362
         w(6) = .313706645877887
         w(7) = .222381034453374
         w(8) = .101228536290376
c
         ra(1)=-.960289856497536
         ra(2)=-.796666477413627
         ra(3)=-.525532409916329
         ra(4)=-.183434642495650
         ra(5)= .183434642495650
         ra(6)= .525532409916329
         ra(7)= .796666477413627
         ra(8)= .960289856497536
        endif

      if (nen.eq.1) xa(1) = zero
c
      if (nen.eq.2) then
         xa(1) = -one
         xa(2) =  one
      endif
c
      if(nen.eq.3) then
         xa(1)= -one
         xa(2)= one
         xa(3)= zero
      endif
c
      if (nen.eq.4) then
         xa(1) = -one
         xa(2) = one
         xa(3) = -.333333333333333
         xa(4) =  .333333333333333
         endif
c
       if(nen.eq.5) then
         xa(1)= -one
         xa(2)=  one
         xa(3)= -pt5
         xa(4)= zero
         xa(5)= pt5
       endif
c
        if(nen.eq.6) then
         xa(1) = -one
         xa(2) =  one
         xa(3) = -.600000000000000
         xa(4) = -.200000000000000
         xa(5) =  .200000000000000
         xa(6) =  .600000000000000
        endif
c
        if(nen.eq.7) then
         xa(1) = -one
         xa(2) =  one
         xa(3) = -.666666666666666
         xa(4) = -.333333333333333
         xa(5) = zero
         xa(6) =  .333333333333333
         xa(7) =  .666666666666666
        endif

        if(nen.eq.8) then
         xa(1) = -one
         xa(2) =  one
         xa(3) = -0.71428571428571
         xa(4) = -0.42857142857143
         xa(5) = -0.14285714285714
         xa(6) = 0.14285714285714
         xa(7) = 0.42857142857143
         xa(8) = 0.71428571428571
        endif
c
      do 100 l = 1, nint
         r = ra(l)
c
        if(nen.eq.1) then
        shl(1,1,l) = zero
        shl(2,1,l) = one
        go to 100
        endif
c
        do 50 i = 1, nen
         aa = one
         bb = one
         aax = zero
         do 40 j =1, nen
          daj = one
          if (i .ne. j)then
           aa = aa * ( r - xa(j))
           bb = bb * ( xa(i) - xa(j))
           do 30 k = 1, nen
            if(k .ne. i .and. k .ne. j) then
              daj = daj * ( r - xa(k))
            endif
   30      continue
           aax =aax + daj
          endif
   40    continue
        shl(2,i,l) = aa/bb
        shl(1,i,l) = aax/bb
   50  continue
c
  100  continue
       return
       end

c-----------------------------------------------------------------------
      subroutine oneshg(xl,det,shl,shg,nen,nint,nesd,ns,nel,neg)
c-----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions
c        and jacobian determinants for the bi-dimensional,
c        elastic beam element
c           xl(j,l) = global coordinates of integration points
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = shape function
c        shg(1,i,l) = global ("arc-length") derivative of shape ftn
c        shg(2,i,l) = shl(2,i,l)
c            det(l) = euclidean length
c                 i = local node number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension xl(nesd,*),det(*),shl(3,nen,*),shg(3,nen,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do l=1,nint
c
         det(l)=zero
         x1=0.d0
         x2=0.d0
         do j=1,nen
            x1=x1+shl(1,j,l)*xl(1,j)
            x2=x2+shl(1,j,l)*xl(2,j)
         end do
         det(l)=dsqrt(x1*x1+x2*x2)
c
         if (det(l).le.zero) then
            write(ieco,1000) ns,nel,neg
            stop
         endif
c
         do i=1,nen
            shg(1,i,l)=shl(1,i,l)/det(l)
            shg(2,i,l)=shl(2,i,l)
         end do
c
      end do
c
 1000 format(///,' oneshg - non-positive determinant in side ',i10,/,
     &     ' in element ',i10,5x,' in element group  ',i10)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine oneshgp(xl,det,shl,shlp,shgp,
     &     nen,npars,nint,nesd,ns,nel,neg)
c-----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions
c        and jacobian determinants for the bi-dimensional,
c        elastic beam element
c
c           xl(j,l) = global coordinates of integration points
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = shape function
c        shg(1,i,l) = global ("arc-length") derivative of shape ftn
c        shg(2,i,l) = shl(2,i,l)
c            det(l) = euclidean length
c                 i = local node number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension xl(nesd,*),det(*),shl(3,nen,*)
      dimension shlp(3,npars,*),shgp(3,npars,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do l=1,nint
c
         det(l)=zero
         x1=0.d0
         x2=0.d0
         do j=1,nen
            x1=x1+shl(1,j,l)*xl(1,j)
            x2=x2+shl(1,j,l)*xl(2,j)
         end do
         det(l)=dsqrt(x1*x1+x2*x2)
c
         if (det(l).le.zero) then
            write(iecho,1000) ns,nel,neg
            stop
         endif
c
         do i=1,npars
            shgp(1,i,l)=shlp(1,i,l)/det(l)
            shgp(2,i,l)=shlp(2,i,l)
         end do
c
      end do
c
 1000 format(///,' oneshgp - non-positive determinant in side ',i10,/,
     &     ' in element ',i10,5x,'in oneshgp group  ',i10)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine twoshgp(xl,det,shl,shlp,shgp,
     &                   nen,npars,nint,nesd,ns,nel,neg,xn)
c-----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions
c     and jacobian determinants for the bi-dimensional (face of element)
c
c     xl(j,l) = global coordinates of integration points
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = shape function
c        shg(1,i,l) = global ("arc-length") derivative of shape ftn
c        shg(2,i,l) = shl(2,i,l)
c            det(l) = euclidean length
c                 i = local node number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension xl(nesd,*),det(*),shl(3,nen,*)
      dimension shlp(3,npars,*),shgp(3,npars,*)
      dimension xst(2,3),xs(3,2),va(3),vb(3),vc(3),xn(3)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
c     TODO: preciso conferir aqui...mas vai ser um mapeamento 2d
c     conforme o codigo abaixo
c
c     ADAPTACAO da subroutine oneshgp
c
      call move(shgp,shlp,3*npars*nint)
c
c$$$      write(*,*) "TWOSHGP"
c$$$      write(*,*) "nesd", nesd, "nen", nen, "npars", npars
c$$$      do i=1,nen
c$$$         write(*,*) xl(1,i), xl(2,i), xl(3,i)
c$$$      end do
c$$$      write(*,*) "SHLP deriv 1"
c$$$      do l=1,nint
c$$$         write(*,*) (shlp(1,i,l), i=1,npars)
c$$$      end do
c$$$      write(*,*) "SHLP deriv 2"
c$$$      do l=1,nint
c$$$         write(*,*) (shlp(2,i,l), i=1,npars)
c$$$      end do
c$$$      write(*,*) "SHLP func 3"
c$$$      do l=1,nint
c$$$         write(*,*) (shlp(3,i,l), i=1,npars)
c$$$      end do
c
      do l=1,nint
c
         do j=1,3
            do i=1,2
               xst(i,j) = rowdot(shl(i,1,l),xl(j,1),3,3,nen)
            end do
         end do
c
c$$$         write(*,*) "MATRIZ XST"
c$$$         do i=1,2
c$$$            write(*,*) xst(i,1), xst(i,2), xst(i,3)
c$$$         end do
c
         do j=1,3
            do i=1,2
               xs(j,i) = xst(i,j)
            end do
            va(j) = xs(j,1)
            vb(j) = xs(j,2)
         end do
c
         call cross(va,vb,vc)
         det(l) = sqrt(vc(1)*vc(1) + vc(2)*vc(2) + vc(3)*vc(3))
c
c      write(*,*) "det_twoshgp = ", det(l)
c
         if (det(l).le.zero) then
            write(ieco,1000) nel,neg
            write(*,1000) nel,neg
            stop
         endif
c
         xn(1) = vc(1)/det(l)
         xn(2) = vc(2)/det(l)
         xn(3) = vc(3)/det(l)
c
c     TODO: preciso verificar essa parte
c           para shgp(1,,) e shgp(2,,) apenas
c
         do i=1,npars
c            shgp(1,i,l) = shlp(1,i,l)/det(l)
c            shgp(2,i,l) = shlp(2,i,l)/det(l)
            shgp(1,i,l) = 0.d0
            shgp(2,i,l) = 0.d0
c
c     para funcao OK
c
            shgp(3,i,l) = shlp(3,i,l)
         end do
      end do
c
 1000 format(///,' twoshgp - non-positive determinant in side ',i10,/,
     &          ' in element ',i10,5x,'in twoshgp group  ',i10)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shgq(xl,det,shl,shg,nint,nel,neg,quad,nen)
c-----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions and
c        jacobian determinants for a  quadrilateral element
c
c        xl(j,i)    = global coordinates
c        det(l)     = jacobian determinant
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c        shg(1,i,l) = x-derivative of shape function
c        shg(2,i,l) = y-derivative of shape function
c        shg(3,i,l) = shl(3,i,l)
c        xs(i,j)    = jacobian matrix
c                 i = local node number or global coordinate number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points, eq. 1 or 4
c
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical quad
      dimension xl(2,*),det(*),shl(3,nen,*),shg(3,nen,*),xs(2,2)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      call move(shg,shl,3*nen*nint)
c
      do 700 l=1,nint
c
      if (.not.quad) then
         do 100 i=1,3
         shg(i,3,l) = shl(i,3,l) + shl(i,4,l)
         shg(i,4,l) = zero
  100    continue
      endif
c
      do 300 j=1,2
      do 200 i=1,2
      xs(i,j) = rowdot(shg(i,1,l),xl(j,1),3,2,nen)
  200 continue
  300 continue
c
      det(l) = xs(1,1)*xs(2,2)-xs(1,2)*xs(2,1)
      if (det(l).le.zero) then
         write(ieco,1000) nel,neg
         stop
      endif
c
      do 500 j=1,2
      do 400 i=1,2
      xs(i,j) = xs(i,j)/det(l)
  400 continue
  500 continue
c
      do 600 i=1,nen
        temp = xs(2,2)*shg(1,i,l) - xs(1,2)*shg(2,i,l)
        shg(2,i,l) = - xs(2,1)*shg(1,i,l) + xs(1,1)*shg(2,i,l)
        shg(1,i,l) = temp
  600 continue
c
  700 continue
c
      return
c
 1000 format(////,' shgq - non-positive determinant - element ',i10,
     &          ' in element group  ',i10)
      end

c-----------------------------------------------------------------------
      subroutine shltpbk(shl,nen,nside,nints)
c-----------------------------------------------------------------------
c     Objetivo: calcular pesos, funcoes de interpolacao e derivadas
c               locais para elementos triangulares
c----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a triangular element
c
c        c1, c2, c3 = local element coordinates ("l1", "l2", "l3".)
c        shl(j,i,l) = local ("j") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension shl(3,nen,*),cl1(18),cl2(18),cl3(18),ra(8)
c
      data   zero,pt1667,pt25,pt5
     &      /0.0d0,0.1666666666666667d0,0.25d0,0.5d0/,
     &       one,two,three,four,five,six
     &      /1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0/
c
      pt3s2 = three/two
      pt9s2 = 9.d0/two
      pt27s2= 27.d0/two
      pt2s3 = two/three
      pt1s6 = one/six
      pt32s3= 32.d0/three
      pt8s3 = 8.d0/three
c
      if (nints.eq.1) then
         ra(1) = zero
      endif
c
      if (nints.eq.2) then
         ra(1)=-.577350269189626
         ra(2)= .577350269189625
      endif
c
      if (nints.eq.3) then
         ra(1)=-.774596669241483
         ra(2)= zero
         ra(3)= .774596669241483
      endif
c
      if (nints.eq.4) then
         ra(1)=-.861136311594053
         ra(2)=-.339981043584856
         ra(3)= .339981043584856
         ra(4)= .861136311594053
      endif
c
       if(nints.eq.5) then
        ra(1)=-.906179845938664
        ra(2)=-.538469310105683
        ra(3)= zero
        ra(4)= .538469310105683
        ra(5)= .906179845938664
       endif
c
       if(nints.eq.6) then
         ra(1)=-.932469514203152
         ra(2)=-.661209386466265
         ra(3)=-.238619186083197
         ra(4)= .238619186083197
         ra(5)= .661209386466365
         ra(6)= .932469514203152
        endif
c
       if(nints.eq.7) then
         ra(1)=-.949107912342759
         ra(2)=-.741531185599394
         ra(3)=-.405845151377397
         ra(4)= zero
         ra(5)= .405845151377397
         ra(6)= .741531185599394
         ra(7)= .949107912342759
        endif

c
      lb = 0
      do 100 ns=1,3
      do 200 ls=1,nints
c
      lb = lb + 1
c
      if(ns.eq.1) then
        cl3(ls) = 0.d00
        cl1(ls) = (1.d00 - ra(ls))/two
        cl2(ls) = (1.d00 + ra(ls))/two
      end if
c
      if(ns.eq.2) then
         cl1(ls) = 0.d00
         cl2(ls) = (1.d00 - ra(ls))/two
         cl3(ls) = (1.d00 + ra(ls))/two
      end if
c
      if(ns.eq.3) then
         cl2(ls) = 0.d00
         cl3(ls) = (1.d00 - ra(ls))/two
         cl1(ls) = (1.d00 + ra(ls))/two
      end if
c
            c1 = cl1(ls)
            c2 = cl2(ls)
            c3 = cl3(ls)
c
            if(nen.eq.1) then
              shl(1,1,lb)= zero
              shl(2,1,lb)= zero
              shl(3,1,lb)= one
            end if
!          acrescentei o if (2012-10-08)
!           interpolação linear (p=1 e nen=3)(2012-10-08)
            if(nen.eq.3) then
              shl(1,1,lb)= one
              shl(2,1,lb)= zero
              shl(3,1,lb)= c1
              shl(1,2,lb)= zero
              shl(2,2,lb)= one
              shl(3,2,lb)= c2
              shl(1,3,lb)=-one
              shl(2,3,lb)=-one
              shl(3,3,lb)= c3
            end if
!           interpolação quadrática (p=2 e nen=6)(2012-10-08)
            if(nen.eq.6) then
              shl(1,1,lb)= four*c1-one
              shl(2,1,lb)= zero
              shl(3,1,lb)= (two*c1 - one)*c1
              shl(1,2,lb)= zero
              shl(2,2,lb)= four*c2-one
              shl(3,2,lb)= (two*c2 - one)*c2
              shl(1,3,lb)= one - four*c3
              shl(2,3,lb)= one - four*c3
              shl(3,3,lb)= (two*c3 - one)*c3
              shl(1,4,lb)= four * c2
              shl(2,4,lb)= four * c1
              shl(3,4,lb)= four * c1 * c2
              shl(1,5,lb)=-four * c2
              shl(2,5,lb)= four * (c3 - c2)
              shl(3,5,lb)= four * c2 * c3
              shl(1,6,lb)= four * (c3 - c1)
              shl(2,6,lb)=-four * c1
              shl(3,6,lb)= four * c3 * c1
            end if
c
            if (nen.eq.10) then

                  shl(1,1,lb)= pt5*((three*c1-one)*(three*c1-two)
     &            +c1*three*(three*c1-two)+c1*three*(three*c1-one))
                  shl(2,1,lb)= zero
                  shl(3,1,lb)= pt5*c1*(three*c1-one)*(three*c1-two)

                  shl(1,2,lb)=zero
                  shl(2,2,lb)=  pt5*((three*c2-one)*(three*c2-two)
     &            +c2*three*(three*c2-two)+c2*three*(three*c2-one))
                  shl(3,2,lb)=  pt5*c2*(three*c2-one)*(three*c2-two)

                  shl(1,3,lb)= -pt5*((three*c3-one)*(three*c3-two)
     &            +c3*three*(three*c3-two)+c3*three*(three*c3-one))
                  shl(2,3,lb)=-pt5*((three*c3-one)*(three*c3-two)
     &            +c3*three*(three*c3-two)+c3*three*(three*c3-one))
                  shl(3,3,lb)=  pt5*c3*(three*c3-one)*(three*c3-two)


                  shl(1,4,lb) = (9.0d0/two)*(c2*(three*c1-one)
     &            +c1*c2*three)
                  shl(2,4,lb) = (9.0d0/two)*c1*(three*c1-one)
                  shl(3,4,lb) = (9.0d0/two)*c1*c2*(three*c1-one)

                  shl(1,5,lb) = (9.0d0/two)*c2*(three*c2-one)
                  shl(2,5,lb) = (9.0d0/two)*(c1*(three*c2-one)
     &            +c1*c2*three)
                  shl(3,5,lb) = (9.0d0/two)*c1*c2*(three*c2-one)


                 shl(1,6,lb) =- (9.0d0/two)*c2*(three*c2-one)
                 shl(2,6,lb) =  (9.0d0/two)*(c3*(three*c2-one)
     &           +c2*c3*three-c2*(three*c2-one))
                 shl(3,6,lb) = (9.0d0/two)*c3*c2*(three*c2-one)


                  shl(1,7,lb) = (9.0d0/two)*(-c2*(three*c3-one)
     &            -c2*c3*three)
                  shl(2,7,lb) =  (9.0d0/two)*(c3*(three*c3-one)
     &            -c2*c3*three -c2*(three*c3-one))
                  shl(3,7,lb) = (9.0d0/two)*c2*c3*(three*c3-one)

                  shl(1,9,lb)= (9.0d0/two)*(-c1*(three*c1-one)
     &            +c1*c3*three +c3*(three*c1-one))
                  shl(2,9,lb) =  (9.0d0/two)*(-c1*(three*c1-one))
                  shl(3,9,lb) = (9.0d0/two)*c3*c1*(three*c1-one)


                  shl(1,8,lb) = (9.0d0/two)*(-c1*(three*c3-one)
     &            -c1*c3*three +c3*(three*c3-one))
                  shl(2,8,lb) =  (9.0d0/two)*(
     &            -c1*c3*three -c1*(three*c3-one))
                  shl(3,8,lb) = (9.0d0/two)*c1*c3*(three*c3-one)

                 shl(1,10,lb) = 27.0d0*(c2*c3-c1*c2)
                 shl(2,10,lb) = 27.0d0*(c1*c3-c1*c2)
                 shl(3,10,lb) = 27.0d0*c1*c2*c3
          endif
!           interpolação quártica (p=4 e nen=15)(2012-10-23)
            if(nen.eq.15) then

              shl(1,1,lb)= pt2s3*(four*c1-two)*(four*c1-one)*c1
     &                  + pt2s3*(four*c1-three)*(four*c1-one)*c1
     &                  + pt2s3*(four*c1-three)*(four*c1-two)*c1
     &            + pt1s6*(four*c1-three)*(four*c1-two)*(four*c1-one)
              shl(2,1,lb)= zero
              shl(3,1,lb)= pt1s6*(four*c1-three)*(four*c1-two)*
     &                  (four*c1-one)*c1

              shl(1,2,lb)= zero
              shl(2,2,lb)= pt2s3*(four*c2-two)*(four*c2-one)*c2
     &                  + pt2s3*(four*c2-three)*(four*c2-one)*c2
     &                  + pt2s3*(four*c2-three)*(four*c2-two)*c2
     &            + pt1s6*(four*c2-three)*(four*c2-two)*(four*c2-one)
              shl(3,2,lb)= pt1s6*(four*c2-three)*(four*c2-two)*
     &                  (four*c2-one)*c2

              shl(1,3,lb)= -pt2s3*(four*c3-two)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-two)*c3
     &             -pt1s6*(four*c3-three)*(four*c3-two)*(four*c3-one)
              shl(2,3,lb)= -pt2s3*(four*c3-two)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-two)*c3
     &             -pt1s6*(four*c3-three)*(four*c3-two)*(four*c3-one)
              shl(3,3,lb)= pt1s6*(four*c3-three)*(four*c3-two)*
     &                   (four*c3-one)*c3

              shl(1,4,lb)= pt32s3*c2*(four*c1-one)*c1
     &                  +pt32s3*c2*(four*c1-two)*c1
     &                  +pt8s3*c2*(four*c1-two)*(four*c1-one)
              shl(2,4,lb)= pt8s3*(four*c1-two)*(four*c1-one)*c1
              shl(3,4,lb)= pt8s3*c2*(four*c1-two)*(four*c1-one)*c1

              shl(1,5,lb)= 16.d0*c1*(four*c2-one)*c2
     &                  + four*(four*c1-one)*(four*c2-one)*c2
              shl(2,5,lb)= 16.d0*c2*(four*c1-one)*c1
     &                  + four*(four*c1-one)*c1*(four*c2-one)
              shl(3,5,lb)= four*(four*c1-one)*c1*(four*c2-one)*c2

              shl(1,6,lb)= pt8s3*(four*c2-two)*(four*c2-one)*c2
              shl(2,6,lb)= pt32s3*c1*(four*c2-one)*c2
     &                  + pt32s3*c1*(four*c2-two)*c2
     &                  + pt8s3*c1*(four*c2-two)*(four*c2-one)
              shl(3,6,lb)= pt8s3*c1*(four*c2-two)*(four*c2-one)*c2

              shl(1,7,lb)= -pt8s3*(four*c2-two)*(four*c2-one)*c2
              shl(2,7,lb)= -pt8s3*(four*c2-two)*(four*c2-one)*c2
     &                  + pt32s3*c3*(four*c2-one)*c2
     &                  + pt32s3*c3*(four*c2-two)*c2
     &                  + pt8s3*c3*(four*c2-two)*(four*c2-one)
              shl(3,7,lb)= pt8s3*c3*(four*c2-two)*(four*c2-one)*c2

              shl(1,8,lb)= -16.d0*(four*c2-one)*c2*c3
     &                  -four*(four*c2-one)*c2*(four*c3-one)
              shl(2,8,lb)= 16.d0*c2*(four*c3-one)*c3
     &                  +four*(four*c2-one)*(four*c3-one)*c3
     &                  -16.d0*(four*c2-one)*c2*c3
     &                  -four*(four*c2-one)*c2*(four*c3-one)
              shl(3,8,lb)= four*(four*c2-one)*c2*(four*c3-one)*c3

              shl(1,9,lb)= -pt32s3*c2*(four*c3-one)*c3
     &                  -pt32s3*c2*(four*c3-two)*c3
     &                  -pt8s3*c2*(four*c3-two)*(four*c3-one)
              shl(2,9,lb)= pt8s3*(four*c3-two)*(four*c3-one)*c3
     &                  -pt32s3*c2*(four*c3-one)*c3
     &                  -pt32s3*c2*(four*c3-two)*c3
     &                  -pt8s3*c2*(four*c3-two)*(four*c3-one)
              shl(3,9,lb)= pt8s3*c2*(four*c3-two)*(four*c3-one)*c3

              shl(1,10,lb)= pt8s3*(four*c3-two)*(four*c3-one)*c3
     &                   -pt32s3*c1*(four*c3-one)*c3
     &                   -pt32s3*c1*(four*c3-two)*c3
     &                   -pt8s3*c1*(four*c3-two)*(four*c3-one)
              shl(2,10,lb)= -pt32s3*c1*(four*c3-one)*c3
     &                   -pt32s3*c1*(four*c3-two)*c3
     &                   -pt8s3*c1*(four*c3-two)*(four*c3-one)
              shl(3,10,lb)= pt8s3*c1*(four*c3-two)*(four*c3-one)*c3

              shl(1,11,lb)= -16.d0*c3*(four*c1-one)*c1
     &                   -4.d0*(four*c3-one)*(four*c1-one)*c1
     &                   +16.d0*c1*(four*c3-one)*c3
     &                   +4.d0*(four*c3-one)*c3*(four*c1-one)
              shl(2,11,lb)= -16.d0*c3*(four*c1-one)*c1
     &                   -4.d0*(four*c3-one)*(four*c1-one)*c1
              shl(3,11,lb)= 4.d0*(four*c3-one)*c3*(four*c1-one)*c1

              shl(1,12,lb)= -pt8s3*(four*c1-two)*(four*c1-one)*c1
     &                   +pt32s3*c3*(four*c1-one)*c1
     &                   +pt32s3*c3*(four*c1-two)*c1
     &                   +pt8s3*c3*(four*c1-two)*(four*c1-one)
              shl(2,12,lb)= -pt8s3*(four*c1-two)*(four*c1-one)*c1
              shl(3,12,lb)= pt8s3*c3*(four*c1-two)*(four*c1-one)*c1

              shl(1,13,lb)= 32.d0*c2*c3*(four*c1-one)
     &                   -32.d0*c1*c2*(four*c1-one)+128.d0*c1*c2*c3
              shl(2,13,lb)= 32.d0*c1*c3*(four*c1-one)
     &                   -32.d0*c1*c2*(four*c1-one)
              shl(3,13,lb)= 32.d0*c1*c2*c3*(four*c1-one)

              shl(1,14,lb)= 32.d0*(four*c2-one)*c2*c3
     &                   -32.d0*c1*(four*c2-one)*c2
              shl(2,14,lb)= 32.d0*c1*c3*(four*c2-one)
     &                   -32.d0*c1*c2*(four*c2-one)+128.d0*c1*c2*c3
              shl(3,14,lb)= 32.d0*c1*c2*c3*(four*c2-one)

              shl(1,15,lb)= 32.d0*c2*(four*c3-one)*c3
     &                   -32.d0*c1*c2*(four*c3-one)-128.d0*c1*c2*c3
              shl(2,15,lb)= 32.d0*c1*(four*c3-one)*c3
     &                   -32.d0*c1*c2*(four*c3-one)-128.d0*c1*c2*c3
              shl(3,15,lb)= 32.d0*c1*c2*c3*(four*c3-one)

            end if
!           interpolação quíntica (p=5 e nen=21)(2012-10-24)
            if(nen.eq.21) then

        shl(1,1,lb)=
     &     (five/24.d0)*(five*c1-three)*(five*c1-two)*(five*c1-one)*c1
     &    +(five/24.d0)*(five*c1-four)*(five*c1-two)*(five*c1-one)*c1
     &    +(five/24.d0)*(five*c1-four)*(five*c1-three)*(five*c1-one)*c1
     &    +(five/24.d0)*(five*c1-four)*(five*c1-three)*(five*c1-two)*c1
     &    +(one/24.d0)*(five*c1-four)*(five*c1-three)
     &                *(five*c1-two)*(five*c1-one)
        shl(2,1,lb)= zero
        shl(3,1,lb)= (one/24.d0)*(five*c1-four)*(five*c1-three)
     &                    *(five*c1-two)*(five*c1-one)*c1

        shl(1,2,lb)= zero
        shl(2,2,lb)=
     &     (five/24.d0)*(five*c2-three)*(five*c2-two)*(five*c2-one)*c2
     &    +(five/24.d0)*(five*c2-four)*(five*c2-two)*(five*c2-one)*c2
     &    +(five/24.d0)*(five*c2-four)*(five*c2-three)*(five*c2-one)*c2
     &    +(five/24.d0)*(five*c2-four)*(five*c2-three)*(five*c2-two)*c2
     &    +(one/24.d0)*(five*c2-four)*(five*c2-three)
     &                *(five*c2-two)*(five*c2-one)
        shl(3,2,lb)= (one/24.d0)*(five*c2-four)*(five*c2-three)
     &                    *(five*c2-two)*(five*c2-one)*c2

        shl(1,3,lb)=
     &    -(five/24.d0)*(five*c3-three)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-two)*c3
     &    -(one/24.d0)*(five*c3-four)*(five*c3-three)
     &                 *(five*c3-two)*(five*c3-one)
        shl(2,3,lb)=
     &    -(five/24.d0)*(five*c3-three)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-two)*c3
     &    -(one/24.d0)*(five*c3-four)*(five*c3-three)
     &                  *(five*c3-two)*(five*c3-one)
        shl(3,3,lb)= (one/24.d0)*(five*c3-four)*(five*c3-three)
     &                    *(five*c3-two)*(five*c3-one)*c3

        shl(1,4,lb)= (125.d0/24.d0)*(five*c1-two)*(five*c1-one)*c1*c2
     &   + (125.d0/24.d0)*(five*c1-three)*(five*c1-one)*c1*c2
     &   + (125.d0/24.d0)*(five*c1-three)*(five*c1-two)*c1*c2
     &   + (25.d0/24.d0)*(five*c1-three)*(five*c1-two)
     &                  *(five*c1-one)*c2
        shl(2,4,lb)= (25.d0/24.d0)*(five*c1-three)
     &                    *(five*c1-two)*(five*c1-one)*c1
        shl(3,4,lb)= (25.d0/24.d0)*(five*c1-three)
     &                    *(five*c1-two)*(five*c1-one)*c1*c2

        shl(1,5,lb)= (125.d0/12.d0)*(five*c1-one)*c1*(five*c2-one)*c2
     &    +(125.d0/12.d0)*(five*c1-two)*c1*(five*c2-one)*c2
     &    +(25.d0/12.d0)*(five*c1-two)*(five*c1-one)*(five*c2-one)*c2
        shl(2,5,lb)= (125.d0/12.d0)*(five*c1-two)*(five*c1-one)*c1*c2
     &    +(25.d0/12.d0)*(five*c1-two)*(five*c1-one)*c1*(five*c2-one)
        shl(3,5,lb)= (25.d0/12.d0)*(five*c1-two)
     &                    *(five*c1-one)*c1*(five*c2-one)*c2

        shl(1,6,lb)= (125.d0/12.d0)*c1*(five*c2-two)*(five*c2-one)*c2
     &   + (25.d0/12.d0)*(five*c1-one)*(five*c2-two)*(five*c2-one)*c2
        shl(2,6,lb)= (125.d0/12.d0)*(five*c1-one)*c1*(five*c2-one)*c2
     &   + (125.d0/12.d0)*(five*c1-one)*c1*(five*c2-two)*c2
     &   + (25.d0/12.d0)*(five*c1-one)*c1*(five*c2-two)*(five*c2-one)
        shl(3,6,lb)= (25.d0/12.d0)*(five*c1-one)
     &                    *c1*(five*c2-two)*(five*c2-one)*c2

        shl(1,7,lb)= (25.d0/24.d0)*(five*c2-three)
     &                    *(five*c2-two)*(five*c2-one)*c2
        shl(2,7,lb)= (125.d0/24.d0)*c1*(five*c2-two)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c1*(five*c2-three)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c1*(five*c2-three)*(five*c2-two)*c2
     &    +(25.d0/24.d0)*c1*(five*c2-three)*(five*c2-two)*(five*c2-one)
        shl(3,7,lb)= (25.d0/24.d0)*c1*
     &                (five*c2-three)*(five*c2-two)*(five*c2-one)*c2

        shl(1,8,lb)= -(25.d0/24.d0)*(five*c2-three)
     &                     *(five*c2-two)*(five*c2-one)*c2
        shl(2,8,lb)=
     &    -(25.d0/24.d0)*(five*c2-three)*(five*c2-two)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c3*(five*c2-two)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c3*(five*c2-three)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c3*(five*c2-three)*(five*c2-two)*c2
     &    +(25.d0/24.d0)*c3*(five*c2-three)*(five*c2-two)*(five*c2-one)
        shl(3,8,lb)= (25.d0/24.d0)*c3*
     &                 (five*c2-three)*(five*c2-two)*(five*c2-one)*c2

        shl(1,9,lb)= - (125.d0/12.d0)*c3*(five*c2-two)*(five*c2-one)*c2
     &   - (25.d0/12.d0)*(five*c3-one)*(five*c2-two)*(five*c2-one)*c2
        shl(2,9,lb)= -((125.d0/12.d0)*c3)*(five*c2-two)*(five*c2-one)*c2
     &   -((25.d0/12.d0)*(five*c3-one))*(five*c2-two)*(five*c2-one)*c2
     &   +((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-one)*c2
     &   +((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-two)*c2
     &   +((25.d0/12.d0)*(five*c3-one))*c3*(five*c2-two)*(five*c2-one)
        shl(3,9,lb)= ((25.d0/12.d0)*(five*c3-one))
     &                     *c3*(five*c2-two)*(five*c2-one)*c2

       shl(1,10,lb)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-one)*c2
     &   -((125.d0/12.d0)*(five*c3-two))*c3*(five*c2-one)*c2
     &   -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c2-one)*c2
       shl(2,10,lb)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-one)*c2
     &    -((125.d0/12.d0)*(five*c3-two))*c3*(five*c2-one)*c2
     &    -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c2-one)*c2
     &    +((125.d0/12.d0)*(five*c3-two))*(five*c3-one)*c3*c2
     &    +((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*c3*(five*c2-one)
        shl(3,10,lb)= ((25.d0/12.d0)*(five*c3-two))*(five*c3-one)
     &                      *c3*(five*c2-one)*c2

       shl(1,11,lb)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c2
     &   -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c2
     &   -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c2
     &  -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c2
       shl(2,11,lb)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c2
     &   -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c2
     &   -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c2
     &   -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c2
     &   +((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c3
        shl(3,11,lb)= ((25.d0/24.d0)*(five*c3-three))
     &                     *(five*c3-two)*(five*c3-one)*c3*c2

       shl(1,12,lb)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c1
     &             -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c1
     &             -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c1
     &   -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c1
     &   +((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c3
       shl(2,12,lb)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c1
     &             -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c1
     &             -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c1
     &   -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c1
              shl(3,12,lb)= ((25.d0/24.d0)*(five*c3-three))
     &                      *(five*c3-two)*(five*c3-one)*c3*c1

       shl(1,13,lb)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-one)*c1
     &           -((125.d0/12.d0)*(five*c3 - two))*c3*(five*c1 - one)*c1
     &     -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c1-one)*c1
     &           +((125.d0/12.d0)*(five*c3 - two))*(five*c3 - one)*c3*c1
     &     +((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*c3*(five*c1-one)
       shl(2,13,lb)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-one)*c1
     &             -((125.d0/12.d0)*(five*c3-two))*c3*(five*c1-one)*c1
     &     -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c1-one)*c1
              shl(3,13,lb)= ((25.d0/12.d0)*(five*c3-two))
     &                      *(five*c3-one)*c3*(five*c1-one)*c1

       shl(1,14,lb)= -((125.d0/12.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &     -((25.d0/12.d0)*(five*c3-one))*(five*c1-two)*(five*c1-one)*c1
     &             +((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-one)*c1
     &             +((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-two)*c1
     &     +((25.d0/12.d0)*(five*c3-one))*c3*(five*c1-two)*(five*c1-one)
       shl(2,14,lb)= -((125.d0/12.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &     -((25.d0/12.d0)*(five*c3-one))*(five*c1-two)*(five*c1-one)*c1
              shl(3,14,lb)= ((25.d0/12.d0)*(five*c3-one))
     &                       *c3*(five*c1-two)*(five*c1-one)*c1

              shl(1,15,lb)=
     &   -((25.d0/24.d0)*(five*c1-three))*(five*c1-two)*(five*c1-one)*c1
     &          +((125.d0/24.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &        +((125.d0/24.d0)*c3)*(five*c1 - three)*(five*c1 - one)*c1
     &        +((125.d0/24.d0)*c3)*(five*c1 - three)*(five*c1 - two)*c1
     &   +((25.d0/24.d0)*c3)*(five*c1-three)*(five*c1-two)*(five*c1-one)

              shl(2,15,lb)= -((25.d0/24.d0)*(five*c1 - three))
     &                       *(five*c1 - two)*(five*c1 - one)*c1
            shl(3,15,lb)= (25.d0/24.d0)*c3*(five*c1-three)*(five*c1-two)
     &         *(five*c1-one)*c1


              shl(1,16,lb)= ((625.d0/6.d0)*(five*c1-one))*c1*c2*c3
     &                   +((625.d0/6.d0)*(five*c1 - two))*c1*c2*c3
     &           +((125.d0/6.d0)*(five*c1 - two))*(five*c1 - one)*c2*c3
     &           -((125.d0/6.d0)*(five*c1 - two))*(five*c1 - one)*c1*c2
        shl(2,16,lb)= ((125.d0/6.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &              -((125.d0/6.d0)*(five*c1 - two))*(five*c1-one)*c1*c2
       shl(3,16,lb)=((125.d0/6.d0)*(five*c1-two))*(five*c1-one)*c1*c2*c3

              shl(1,17,lb)= ((625.d0/4.d0)*(five*c2 - one))*c1*c2*c3
     &           +((125.d0/4.d0)*(five*c1 - one))*(five*c2 - one)*c2*c3
     &            -((125.d0/4.d0)*(five*c1 - one))*c1*(five*c2 - one)*c2
              shl(2,17,lb)= ((625.d0/4.d0)*(five*c1 - one))*c1*c2*c3
     &           +((125.d0/4.d0)*(five*c1 - one))*(five*c2 - one)*c1*c3
     &           -((125.d0/4.d0)*(five*c1 - one))*c1*(five*c2 - one)*c2
       shl(3,17,lb)=((125.d0/4.d0)*(five*c1-one))*(five*c2-one)*c1*c2*c3

       shl(1,18,lb)= ((125.d0/6.d0)*c3)*(five*c2-two)*(five*c2 - one)*c2
     &            -((125.d0/6.d0))*c1*(five*c2 - two)*(five*c2 - one)*c2
              shl(2,18,lb)= ((625.d0/6.d0)*(five*c2 - one))*c1*c2*c3
     &              +((625.d0/6.d0)*(five*c2 - two))*c1*c2*c3
     &           +((125.d0/6.d0)*(five*c2 - two))*(five*c2 - one)*c1*c3
     &           -((125.d0/6.d0))*c1*(five*c2 - two)*(five*c2 - one)*c2
       shl(3,18,lb)=((125.d0/6.d0)*(five*c2-two))*(five*c2-one)*c1*c2*c3

              shl(1,19,lb)= -((625.d0/4.d0)*(five*c2 - one))*c1*c2*c3
     &           +((125.d0/4.d0)*(five*c3 - one))*c3*(five*c2 - one)*c2
     &           -((125.d0/4.d0)*(five*c2 - one))*(five*c3 - one)*c1*c2
              shl(2,19,lb)= ((625.d0/4.d0)*(five*c3 - one))*c1*c2*c3
     &              -((625.d0/4.d0)*(five*c2 - one))*c1*c2*c3
     &           +((125.d0/4.d0)*(five*c2 - one))*(five*c3 - one)*c1*c3
     &           -((125.d0/4.d0)*(five*c2 - one))*(five*c3 - one)*c1*c2
      shl(3,19,lb)= ((125.d0/4.d0)*(five*c2-one))*(five*c3-one)*c1*c2*c3

              shl(1,20,lb)= -((625.d0/6.d0)*(five*c3 - one))*c1*c2*c3
     &              -((625.d0/6.d0)*(five*c3 - two))*c1*c2*c3
     &           +((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c3*c2
     &           -((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c1*c2
              shl(2,20,lb)= -((625.d0/6.d0)*(five*c3 - one))*c1*c2*c3
     &              -((625.d0/6.d0)*(five*c3 - two))*c1*c2*c3
     &           +((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c3*c1
     &           -((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c1*c2
      shl(3,20,lb)= ((125.d0/6.d0)*(five*c3-two))*(five*c3-one)*c1*c2*c3

              shl(1,21,lb)= ((625.d0/4.d0)*(five*c3 - one))*c1*c2*c3
     &              -((625.d0/4.d0)*(five*c1 - one))*c1*c2*c3
     &           +((125.d0/4.d0)*(five*c1 - one))*(five*c3 - one)*c2*c3
     &           -((125.d0/4.d0)*(five*c1 - one))*(five*c3 - one)*c1*c2
              shl(2,21,lb)= -((625.d0/4.d0)*(five*c1 - one))*c1*c2*c3
     &           +((125.d0/4.d0)*(five*c3 - one))*c3*(five*c1 - one)*c1
     &           -((125.d0/4.d0)*(five*c1 - one))*(five*c3 - one)*c1*c2
      shl(3,21,lb)= ((125.d0/4.d0)*(five*c1-one))*(five*c3-one)*c1*c2*c3

            end if

  200     continue
  100     continue
c
c
           return
           end

c----------------------------------------------------------------------
      subroutine shlt(shl,w,nint,nen)
c----------------------------------------------------------------------
c     Objetivo: calcular pesos, funcoes de interpolacao e derivadas locais
c               para elementos triangulares
c----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a triangular element
c        c1, c2, c3 = local element coordinates ("l1", "l2", "l3".)
c        shl(j,i,l) = local ("j") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points,
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension shl(3,nen,*),w(*),cl1(25),cl2(25),cl3(25)
      data   zero,pt1667,pt25,pt5
     &      /0.0d0,0.1666666666666667d0,0.25d0,0.5d0/,
     &       one,two,three,four,five,six
     &      /1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0/
      data r1/0.33333333333333333333d00/,w1/1.d00/,
     &     r2/0.5d00                   /,w2/0.3333333333333333333d00/,
     &     r3a/0.3333333333333333333d00/,w3a/-0.5625d00/,
     &     r3b1/0.6d00                 /,w3b/0.520833333333333333d00/,
     &     r3b2/0.2d00                 /,
     &     r7a/0.3333333333333333333d00/,w7a/0.225d00/,
     &     r7b/0.0597158717d00         /,w7b/0.1323941527d00/,
     &     r7c/0.4701420641d00         /,
     &     r7d/0.7974269853d00         /,w7d/0.1259391805d00/,
     &     r7e/0.1012865073d00         /
      data ri1/0.1666666666666666666d00/,ri2/0.666666666666666666d00/,
     &     ri3/0.1666666666666666666d00/
      data r2a/0.666666666666667d00    /,r2b/ 0.166666666666667d00/,
     &     w6a/0.109951743655322d00    /,w6b/0.223381589678011d00/,
     &     r6a/0.816847572980459d00    /,r6c/0.108103018168070d00/,
     &     r6b/0.091576213509771d00    /,r6d/0.445948490915965d00/,
     &     w12a/0.116786275726379d00   /,
     &     r12a1/0.501426509658179d00  /,r12b1/0.249286745170910d00/,
     &     w12b/0.050844906370207d00   /,
     &     r12a2/0.873821971016996d00  /,r12b2/0.063089014491502d00/,
     &     w12c/0.082851075618374d00   /,
     &     r12a3/0.636502499121399d00  /,r12b3/0.310352451033785d00/,
     &     r12c3/0.053145049844816d00  /,
c
     &     w13a/-0.149570044467670d00  /,w13b/0.175615257433204d00/,
     &     w13c/0.053347235608839d00   /,w13d/0.077113760890257d00 /,
     &     r13a/0.333333333333333d00/,
     &     r13a1/0.479308067841923d00  /,r13b1/0.260345966079038d00/,
     &     r13a2/0.869739794195568d00  /,r13b2/0.065130102902216d00/,
     &     r13a3/0.638444188569809d00  /,r13b3/0.312865496004875d00/,
     &     r13c3/0.048690315425316d00  /
      data w16a/0.144315607677787d00   /,r16a/0.333333333333333d00/,
     &     w16b/0.095091634267285d00   /,
     &     r16b1/0.081414823414554d00  /,r16b2/0.459292588292723d00/,
     &     w16c/0.103217370534718d00   /,
     &     r16c1/0.658861384496480d00  /,r16c2/0.170569307751760d00/,
     &     w16d/0.032458497623198d00   /,
     &     r16d1/0.898905543365938d00  /,r16d2/0.050547228317031d00/,
     &     w16e/0.027230314174435d00   /,r16e1/0.728492392955404d00/,
     &     r16e2/0.263112829634638d00  /,r16e3/0.008394777409958d00/,
c
     &     w19a/0.097135796282799d00   /,r19a/0.333333333333333d00/,
     &     w19b/0.031334700227139d00   /,
     &     r19b1/0.020634961602525d00  /,r19b2/0.489682519198738d00/,
     &     w19c/0.077827541004774d00   /,
     &     r19c1/0.125820817014127d00  /,r19c2/0.437089591492937d00/,
     &     w19d/0.079647738927210d00   /,
     &     r19d1/0.623592928761935d00  /,r19d2/0.188203535619033d00/,
     &     w19e/0.025577675658698d00   /,
     &     r19e1/0.910540973211095d00  /,r19e2/0.044729513394453d00/,
     &     w19f/0.043283539377289d00   /,r19f1/0.741198598784498d00/,
     &     r19f2/0.221962989160766d00  /,r19f3/0.036838412054736d00/,
c
     &     w25a/0.090817990382754d00   /,r25a/0.333333333333333d00/,
     &     w25b/0.036725957756467d00   /,
     &     r25b1/0.028844733232685d00  /,r25b2/0.485577633383657d00/,
     &     w25c/0.045321059435528d00   /,
     &     r25c1/0.781036849029926d00  /,r25c2/0.109481575485037d00/,
     &     w25d/0.072757916845420d00   /,r25d1/0.550352941820999d00/,
     &     r25d2/0.307939838764121d00  /,r25d3/0.141707219414880d00/,
     &     w25e/0.028327242531057d00   /,r25e1/0.728323904597411d00/,
     &     r25e2/0.246672560639903d00  /,r25e3/0.025003534762686d00/,
     &     w25f/0.009421666963733d00   /,r25f1/0.923655933587500d00/,
     &     r25f2/0.066803251012200d00  /,r25f3/0.009540815400299d00/

      pt3s2 = three/two
      pt9s2 = 9.d0/two
      pt27s2= 27.d0/two
      pt2s3 = two/three
      pt1s6 = one/six
      pt32s3= 32.d0/three
      pt8s3 = 8.d0/three
c
      if (nint.eq.1) then
            w(1)=w1/two
            cl1(1)=r1
            cl2(1)=r1
            cl3(1)=one-r1-r1
      end if
c
      if(nint.eq.3) then
        do 31 i=1,3
          w(i)=w2/two
  31    continue
        cl1(1)=r2
        cl2(1)=r2
        cl3(1)=zero
        cl1(2)=zero
        cl2(2)=r2
        cl3(2)=r2
        cl1(3)=r2
        cl2(3)=zero
        cl3(3)=r2
! c  pontos montados nas linhas do meio (artigo Dunavant)
! c           ponto a
!             cl1(1)=r2a
!             cl2(1)=r2b
!             cl3(1)=r2b
! c           ponto b
!             cl1(2)=r2b
!             cl2(2)=r2a
!             cl3(2)=r2b
! c           ponto c
!             cl1(3)=r2b
!             cl2(3)=r2b
!             cl3(3)=r2a
!
      end if
c
      if(nint.eq.4) then
        w(1)= w3a/two
        do 41 i=2,4
          w(i)=w3b/two
  41    continue
        cl1(1)=r3a
        cl2(1)=r3a
        cl3(1)=r3a

        cl1(2)=r3b1
        cl2(2)=r3b2
        cl3(2)=r3b2

        cl1(3)=r3b2
        cl2(3)=r3b1
        cl3(3)=r3b2

        cl1(4)=r3b2
        cl2(4)=r3b2
        cl3(4)=r3b1
      end if
c
      if(nint.eq.6) then
        do 61 i=1,3
          w(i)=w6a/two
  61    continue
        do 62 i=4,6
          w(i)=w6b/two
  62    continue
        do 63 i=1,3
          cl1(i)=r6b
          cl2(i)=r6b
          cl3(i)=r6b
  63    continue
          cl1(1)=r6a
          cl2(2)=r6a
          cl3(3)=r6a
        do 64 i=4,6
          cl1(i)=r6d
          cl2(i)=r6d
          cl3(i)=r6d
  64    continue
          cl1(4)=r6c
          cl2(5)=r6c
          cl3(6)=r6c
      end if
c
      if(nint.eq.7) then
        w(1)= w7a/two
        do 71 i=2,4
          w(i)=w7b/two
  71    continue
        do 72 i=5,7
          w(i)=w7d/two
  72    continue
          cl1(1)=r7a
          cl2(1)=r7a
          cl3(1)=r7a
        do 73 i=2,4
          cl1(i)=r7c
          cl2(i)=r7c
          cl3(i)=r7c
  73    continue
          cl1(2)=r7b
          cl2(3)=r7b
          cl3(4)=r7b
        do 74 i=5,7
          cl1(i)=r7e
          cl2(i)=r7e
          cl3(i)=r7e
  74    continue
          cl1(5)=r7d
          cl2(6)=r7d
          cl3(7)=r7d
      end if
c
      if(nint.eq.12) then
          do 110 i=1,3
            w(i)=w12a/two
  110     continue
          do 120 i=4,6
            w(i)=w12b/two
  120     continue
          do 130 i=7,12
            w(i)=w12c/two
  130     continue
          do 140 i=1,3
            cl1(i)=r12b1
            cl2(i)=r12b1
            cl3(i)=r12b1
  140     continue
          cl1(1)=r12a1
          cl2(2)=r12a1
          cl3(3)=r12a1
          do 150 i=4,6
            cl1(i)=r12b2
            cl2(i)=r12b2
            cl3(i)=r12b2
  150     continue
          cl1(4)=r12a2
          cl2(5)=r12a2
          cl3(6)=r12a2

          cl1(7)=r12a3
          cl2(7)=r12b3
          cl3(7)=r12c3

          cl1(8)=r12c3
          cl2(8)=r12a3
          cl3(8)=r12b3

          cl1(9)=r12b3
          cl2(9)=r12c3
          cl3(9)=r12a3

          cl1(10)=r12a3
          cl2(10)=r12c3
          cl3(10)=r12b3

          cl1(11)=r12b3
          cl2(11)=r12a3
          cl3(11)=r12c3

          cl1(12)=r12c3
          cl2(12)=r12b3
          cl3(12)=r12a3
      end if
cc
      if (nint.eq.13) then
        w(1)= w13a/two
        do 131 i=2,4
          w(i)=w13b/two
  131   continue
        do 132 i=5,7
          w(i)=w13c/two
  132   continue
        do 133 i=8,13
          w(i)=w13d/two
  133   continue
          cl1(1)=r13a
          cl2(1)=r13a
          cl3(1)=r13a
        do 134 i=2,4
          cl1(i)=r13b1
          cl2(i)=r13b1
          cl3(i)=r13b1
  134   continue
          cl1(2)=r13a1
          cl2(3)=r13a1
          cl3(4)=r13a1
        do 135 i=5,7
          cl1(i)=r13b2
          cl2(i)=r13b2
          cl3(i)=r13b2
  135   continue
          cl1(5)=r13a2
          cl2(6)=r13a2
          cl3(7)=r13a2

          cl1(8)=r13a3
          cl2(8)=r13b3
          cl3(8)=r13c3

          cl1(9)=r13c3
          cl2(9)=r13a3
          cl3(9)=r13b3

          cl1(10)=r13b3
          cl2(10)=r13c3
          cl3(10)=r13a3

          cl1(11)=r13a3
          cl2(11)=r13c3
          cl3(11)=r13b3

          cl1(12)=r13b3
          cl2(12)=r13a3
          cl3(12)=r13c3

          cl1(13)=r13c3
          cl2(13)=r13b3
          cl3(13)=r13a3
      endif
c
      if (nint.eq.16) then
        w(1)= w16a/two
        do 161 i=2,4
          w(i)=w16b/two
  161   continue
        do 162 i=5,7
          w(i)=w16c/two
  162   continue
        do 163 i=8,10
          w(i)=w16d/two
  163   continue
        do 164 i=11,16
          w(i)=w16e/two
  164   continue
          cl1(1)=r16a
          cl2(1)=r16a
          cl3(1)=r16a
        do 165 i=2,4
          cl1(i)=r16b2
          cl2(i)=r16b2
          cl3(i)=r16b2
  165   continue
          cl1(2)=r16b1
          cl2(3)=r16b1
          cl3(4)=r16b1
        do 166 i=5,7
          cl1(i)=r16c2
          cl2(i)=r16c2
          cl3(i)=r16c2
  166   continue
          cl1(5)=r16c1
          cl2(6)=r16c1
          cl3(7)=r16c1
        do 167 i=8,10
          cl1(i)=r16d2
          cl2(i)=r16d2
          cl3(i)=r16d2
  167   continue
          cl1(8)=r16d1
          cl2(9)=r16d1
          cl3(10)=r16d1

          cl1(11)=r16e1
          cl2(11)=r16e2
          cl3(11)=r16e3

          cl1(12)=r16e3
          cl2(12)=r16e1
          cl3(12)=r16e2

          cl1(13)=r16e2
          cl2(13)=r16e3
          cl3(13)=r16e1

          cl1(14)=r16e1
          cl2(14)=r16e3
          cl3(14)=r16e2

          cl1(15)=r16e2
          cl2(15)=r16e1
          cl3(15)=r16e3

          cl1(16)=r16e3
          cl2(16)=r16e2
          cl3(16)=r16e1
      endif
c
      if (nint.eq.19) then
        w(1)= w19a/two
        do 191 i=2,4
          w(i)=w19b/two
  191   continue
        do 192 i=5,7
          w(i)=w19c/two
  192   continue
        do 193 i=8,10
          w(i)=w19d/two
  193   continue
        do 194 i=11,13
          w(i)=w19e/two
  194   continue
        do 195 i=14,19
          w(i)=w19f/two
  195   continue
          cl1(1)=r19a
          cl2(1)=r19a
          cl3(1)=r19a
        do 196 i=2,4
          cl1(i)=r19b2
          cl2(i)=r19b2
          cl3(i)=r19b2
  196   continue
          cl1(2)=r19b1
          cl2(3)=r19b1
          cl3(4)=r19b1
        do 197 i=5,7
          cl1(i)=r19c2
          cl2(i)=r19c2
          cl3(i)=r19c2
  197   continue
          cl1(5)=r19c1
          cl2(6)=r19c1
          cl3(7)=r19c1
        do 198 i=8,10
          cl1(i)=r19d2
          cl2(i)=r19d2
          cl3(i)=r19d2
  198   continue
          cl1(8)=r19d1
          cl2(9)=r19d1
          cl3(10)=r19d1
        do 199 i=11,13
          cl1(i)=r19e2
          cl2(i)=r19e2
          cl3(i)=r19e2
  199   continue
          cl1(11)=r19e1
          cl2(12)=r19e1
          cl3(13)=r19e1

          cl1(14)=r19f1
          cl2(14)=r19f2
          cl3(14)=r19f3

          cl1(15)=r19f3
          cl2(15)=r19f1
          cl3(15)=r19f2

          cl1(16)=r19f2
          cl2(16)=r19f3
          cl3(16)=r19f1

          cl1(17)=r19f1
          cl2(17)=r19f3
          cl3(17)=r19f2

          cl1(18)=r19f2
          cl2(18)=r19f1
          cl3(18)=r19f3

          cl1(19)=r19f3
          cl2(19)=r19f2
          cl3(19)=r19f1
      endif
c
      if (nint.eq.25) then
        w(1)= w25a/two
        do 251 i=2,4
          w(i)=w25b/two
  251   continue
        do 252 i=5,7
          w(i)=w25c/two
  252   continue
        do 253 i=8,13
          w(i)=w25d/two
  253   continue
        do 254 i=14,19
          w(i)=w25e/two
  254   continue
        do 255 i=20,25
          w(i)=w25f/two
  255   continue
          cl1(1)=r25a
          cl2(1)=r25a
          cl3(1)=r25a
        do 256 i=2,4
          cl1(i)=r25b2
          cl2(i)=r25b2
          cl3(i)=r25b2
  256   continue
          cl1(2)=r25b1
          cl2(3)=r25b1
          cl3(4)=r25b1
        do 257 i=5,7
          cl1(i)=r25c2
          cl2(i)=r25c2
          cl3(i)=r25c2
  257   continue
          cl1(5)=r25c1
          cl2(6)=r25c1
          cl3(7)=r25c1

          cl1(8)=r25d1
          cl2(8)=r25d2
          cl3(8)=r25d3

          cl1(9)=r25d3
          cl2(9)=r25d1
          cl3(9)=r25d2

          cl1(10)=r25d2
          cl2(10)=r25d3
          cl3(10)=r25d1

          cl1(11)=r25d1
          cl2(11)=r25d3
          cl3(11)=r25d2

          cl1(12)=r25d2
          cl2(12)=r25d1
          cl3(12)=r25d3

          cl1(13)=r25d3
          cl2(13)=r25d2
          cl3(13)=r25d1

          cl1(14)=r25e1
          cl2(14)=r25e2
          cl3(14)=r25e3

          cl1(15)=r25e3
          cl2(15)=r25e1
          cl3(15)=r25e2

          cl1(16)=r25e2
          cl2(16)=r25e3
          cl3(16)=r25e1

          cl1(17)=r25e1
          cl2(17)=r25e3
          cl3(17)=r25e2

          cl1(18)=r25e2
          cl2(18)=r25e1
          cl3(18)=r25e3

          cl1(19)=r25e3
          cl2(19)=r25e2
          cl3(19)=r25e1

          cl1(20)=r25f1
          cl2(20)=r25f2
          cl3(20)=r25f3

          cl1(21)=r25f3
          cl2(21)=r25f1
          cl3(21)=r25f2

          cl1(22)=r25f2
          cl2(22)=r25f3
          cl3(22)=r25f1

          cl1(23)=r25f1
          cl2(23)=r25f3
          cl3(23)=r25f2

          cl1(24)=r25f2
          cl2(24)=r25f1
          cl3(24)=r25f3

          cl1(25)=r25f3
          cl2(25)=r25f2
          cl3(25)=r25f1

      endif

      do 200 l=1,nint
c
            c1 = cl1(l)
            c2 = cl2(l)
            c3 = cl3(l)
!          acrescentei o if (2012-10-08)
!           interpolação linear (p=1 e nen=3)(2012-10-08)
            if(nen.eq.1) then
              shl(1,1,l)= zero
              shl(2,1,l)= zero
              shl(3,1,l)= one
            end if
            if(nen.eq.3) then
              shl(1,1,l)= one
              shl(2,1,l)= zero
              shl(3,1,l)= c1
              shl(1,2,l)= zero
              shl(2,2,l)= one
              shl(3,2,l)= c2
              shl(1,3,l)=-one
              shl(2,3,l)=-one
              shl(3,3,l)= c3
            end if
!           interpolação quadrática (p=2 e nen=6)(2012-10-08)
            if(nen.eq.6) then
              shl(1,1,l)= four*c1-one
              shl(2,1,l)= zero
              shl(3,1,l)= (two*c1 - one)*c1
              shl(1,2,l)= zero
              shl(2,2,l)= four*c2-one
              shl(3,2,l)= (two*c2 - one)*c2
              shl(1,3,l)= one - four*c3
              shl(2,3,l)= one - four*c3
              shl(3,3,l)= (two*c3 - one)*c3
              shl(1,4,l)= four * c2
              shl(2,4,l)= four * c1
              shl(3,4,l)= four * c1 * c2
              shl(1,5,l)=-four * c2
              shl(2,5,l)= four * (c3 - c2)
              shl(3,5,l)= four * c2 * c3
              shl(1,6,l)= four * (c3 - c1)
              shl(2,6,l)=-four * c1
              shl(3,6,l)= four * c3 * c1
            end if
!           interpolação cúbica (p=3 e nen=10)(2012-10-22)
            if(nen.eq.10) then

              shl(1,1,l)= pt5*(three*c1-two)*(three*c1-one) +
     &                pt3s2*c1*(three*c1-one)+pt3s2*c1*(three*c1-two)
              shl(2,1,l)= zero
              shl(3,1,l)= pt5*c1*(three*c1 - two)*(three*c1-one)

              shl(1,2,l)= zero
              shl(2,2,l)= pt5*(three*c2-two)*(three*c2-one) +
     &                pt3s2*c2*(three*c2-one)+pt3s2*c2*(three*c2-two)
              shl(3,2,l)= pt5*c2*(three*c2 - two)*(three*c2-one)

              shl(1,3,l)= -pt5*(three*c3 - two)*(three*c3-one)
     &          - pt3s2*c3*(three*c3-one) - pt3s2*c3*(three*c3 - two)
              shl(2,3,l)= -pt5*(three*c3 - two)*(three*c3-one)
     &          - pt3s2*c3*(three*c3-one) - pt3s2*c3*(three*c3 - two)
              shl(3,3,l)= pt5*c3*(three*c3 - two)*(three*c3-one)

              shl(1,4,l)= pt9s2*c2*(three*c1-one) + pt27s2*c1*c2
              shl(2,4,l)= pt9s2*c1*(three*c1-one)
              shl(3,4,l)= pt9s2*c1*c2*(three*c1-one)

              shl(1,5,l)= pt9s2*c2*(three*c2-one)
              shl(2,5,l)= pt9s2*c1*(three*c2-one) + pt27s2*c1*c2
              shl(3,5,l)= pt9s2*c1*c2*(three*c2-one)

              shl(1,6,l)= -pt9s2*c2*(three*c2-one)
              shl(2,6,l)= pt9s2*c3*(three*c2-one) + pt27s2*c2*c3
     &                  - pt9s2*c2*(three*c2-one)
              shl(3,6,l)= pt9s2*c2*c3*(three*c2-one)

              shl(1,7,l)= -pt9s2*c2*(three*c3-one) -pt27s2*c2*c3
              shl(2,7,l)= pt9s2*c3*(three*c3-one) - pt27s2*c2*c3
     &                  - pt9s2*c2*(three*c3-one)
              shl(3,7,l)= pt9s2*c2*c3*(three*c3-one)

              shl(1,8,l)= -pt9s2*c1*(three*c3-one) -pt27s2*c3*c1
     &                  + pt9s2*c3*(three*c3-one)
              shl(2,8,l)= -pt9s2*c1*(three*c3-one) - pt27s2*c3*c1
              shl(3,8,l)= pt9s2*c3*c1*(three*c3-one)

              shl(1,9,l)= -pt9s2*c1*(three*c1-one) +pt27s2*c3*c1
     &                  + pt9s2*c3*(three*c1-one)
              shl(2,9,l)= -pt9s2*c1*(three*c1-one)
              shl(3,9,l)= pt9s2*c3*c1*(three*c1-one)

              shl(1,10,l)= 27.d0*c2*c3 - 27.d0*c1*c2
              shl(2,10,l)= 27.d0*c3*c1 - 27.d0*c1*c2
              shl(3,10,l)= 27.d0*c1*c2*c3

            end if
!           interpolação quártica (p=4 e nen=15)(2012-10-23)
            if(nen.eq.15) then

              shl(1,1,l)= pt2s3*(four*c1-two)*(four*c1-one)*c1
     &                  + pt2s3*(four*c1-three)*(four*c1-one)*c1
     &                  + pt2s3*(four*c1-three)*(four*c1-two)*c1
     &            + pt1s6*(four*c1-three)*(four*c1-two)*(four*c1-one)
              shl(2,1,l)= zero
              shl(3,1,l)= pt1s6*(four*c1-three)*(four*c1-two)*
     &                  (four*c1-one)*c1

              shl(1,2,l)= zero
              shl(2,2,l)= pt2s3*(four*c2-two)*(four*c2-one)*c2
     &                  + pt2s3*(four*c2-three)*(four*c2-one)*c2
     &                  + pt2s3*(four*c2-three)*(four*c2-two)*c2
     &            + pt1s6*(four*c2-three)*(four*c2-two)*(four*c2-one)
              shl(3,2,l)= pt1s6*(four*c2-three)*(four*c2-two)*
     &                  (four*c2-one)*c2

              shl(1,3,l)= -pt2s3*(four*c3-two)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-two)*c3
     &             -pt1s6*(four*c3-three)*(four*c3-two)*(four*c3-one)
              shl(2,3,l)= -pt2s3*(four*c3-two)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-one)*c3
     &                  -pt2s3*(four*c3-three)*(four*c3-two)*c3
     &             -pt1s6*(four*c3-three)*(four*c3-two)*(four*c3-one)
              shl(3,3,l)= pt1s6*(four*c3-three)*(four*c3-two)*
     &                   (four*c3-one)*c3

              shl(1,4,l)= pt32s3*c2*(four*c1-one)*c1
     &                  +pt32s3*c2*(four*c1-two)*c1
     &                  +pt8s3*c2*(four*c1-two)*(four*c1-one)
              shl(2,4,l)= pt8s3*(four*c1-two)*(four*c1-one)*c1
              shl(3,4,l)= pt8s3*c2*(four*c1-two)*(four*c1-one)*c1

              shl(1,5,l)= 16.d0*c1*(four*c2-one)*c2
     &                  + four*(four*c1-one)*(four*c2-one)*c2
              shl(2,5,l)= 16.d0*c2*(four*c1-one)*c1
     &                  + four*(four*c1-one)*c1*(four*c2-one)
              shl(3,5,l)= four*c2*(four*c2-one)*(four*c1-one)*c1

              shl(1,6,l)= pt8s3*(four*c2-two)*(four*c2-one)*c2
              shl(2,6,l)= pt32s3*c1*(four*c2-one)*c2
     &                  + pt32s3*c1*(four*c2-two)*c2
     &                  + pt8s3*c1*(four*c2-two)*(four*c2-one)
              shl(3,6,l)= pt8s3*c1*(four*c2-two)*(four*c2-one)*c2

              shl(1,7,l)= -pt8s3*(four*c2-two)*(four*c2-one)*c2
              shl(2,7,l)= -pt8s3*(four*c2-two)*(four*c2-one)*c2
     &                  + pt32s3*c3*(four*c2-one)*c2
     &                  + pt32s3*c3*(four*c2-two)*c2
     &                  + pt8s3*c3*(four*c2-two)*(four*c2-one)
              shl(3,7,l)= pt8s3*c3*(four*c2-two)*(four*c2-one)*c2

              shl(1,8,l)= -16.d0*(four*c2-one)*c2*c3
     &                  -four*(four*c2-one)*c2*(four*c3-one)
              shl(2,8,l)= 16.d0*c2*(four*c3-one)*c3
     &                  +four*(four*c2-one)*(four*c3-one)*c3
     &                  -16.d0*(four*c2-one)*c2*c3
     &                  -four*(four*c2-one)*c2*(four*c3-one)
              shl(3,8,l)= four*(four*c2-one)*c2*(four*c3-one)*c3

              shl(1,9,l)= -pt32s3*c2*(four*c3-one)*c3
     &                  -pt32s3*c2*(four*c3-two)*c3
     &                  -pt8s3*c2*(four*c3-two)*(four*c3-one)
              shl(2,9,l)= pt8s3*(four*c3-two)*(four*c3-one)*c3
     &                  -pt32s3*c2*(four*c3-one)*c3
     &                  -pt32s3*c2*(four*c3-two)*c3
     &                  -pt8s3*c2*(four*c3-two)*(four*c3-one)
              shl(3,9,l)= pt8s3*c2*(four*c3-two)*(four*c3-one)*c3

              shl(1,10,l)= pt8s3*(four*c3-two)*(four*c3-one)*c3
     &                   -pt32s3*c1*(four*c3-one)*c3
     &                   -pt32s3*c1*(four*c3-two)*c3
     &                   -pt8s3*c1*(four*c3-two)*(four*c3-one)
              shl(2,10,l)= -pt32s3*c1*(four*c3-one)*c3
     &                   -pt32s3*c1*(four*c3-two)*c3
     &                   -pt8s3*c1*(four*c3-two)*(four*c3-one)
              shl(3,10,l)= pt8s3*c1*(four*c3-two)*(four*c3-one)*c3

              shl(1,11,l)= -16.d0*c3*(four*c1-one)*c1
     &                   -4.d0*(four*c3-one)*(four*c1-one)*c1
     &                   +16.d0*c1*(four*c3-one)*c3
     &                   +4.d0*(four*c3-one)*c3*(four*c1-one)
              shl(2,11,l)= -16.d0*c3*(four*c1-one)*c1
     &                   -4.d0*(four*c3-one)*(four*c1-one)*c1
              shl(3,11,l)= 4.d0*(four*c3-one)*c3*(four*c1-one)*c1

              shl(1,12,l)= -pt8s3*(four*c1-two)*(four*c1-one)*c1
     &                   +pt32s3*c3*(four*c1-one)*c1
     &                   +pt32s3*c3*(four*c1-two)*c1
     &                   +pt8s3*c3*(four*c1-two)*(four*c1-one)
              shl(2,12,l)= -pt8s3*(four*c1-two)*(four*c1-one)*c1
              shl(3,12,l)= pt8s3*c3*(four*c1-two)*(four*c1-one)*c1

              shl(1,13,l)= 32.d0*c2*c3*(four*c1-one)
     &                   -32.d0*c1*c2*(four*c1-one)+128.d0*c1*c2*c3
              shl(2,13,l)= 32.d0*c1*c3*(four*c1-one)
     &                   -32.d0*c1*c2*(four*c1-one)
              shl(3,13,l)= 32.d0*c1*c2*c3*(four*c1-one)

              shl(1,14,l)= 32.d0*(four*c2-one)*c2*c3
     &                   -32.d0*c1*(four*c2-one)*c2
              shl(2,14,l)= 32.d0*c1*c3*(four*c2-one)
     &                   -32.d0*c1*c2*(four*c2-one)+128.d0*c1*c2*c3
              shl(3,14,l)= 32.d0*c1*c2*c3*(four*c2-one)

              shl(1,15,l)= 32.d0*c2*(four*c3-one)*c3
     &                   -32.d0*c1*c2*(four*c3-one)-128.d0*c1*c2*c3
              shl(2,15,l)= 32.d0*c1*(four*c3-one)*c3
     &                   -32.d0*c1*c2*(four*c3-one)-128.d0*c1*c2*c3
              shl(3,15,l)= 32.d0*c1*c2*c3*(four*c3-one)

            end if
!           interpolação quíntica (p=5 e nen=21)(2012-10-24)
            if(nen.eq.21) then

        shl(1,1,l)=
     &     (five/24.d0)*(five*c1-three)*(five*c1-two)*(five*c1-one)*c1
     &    +(five/24.d0)*(five*c1-four)*(five*c1-two)*(five*c1-one)*c1
     &    +(five/24.d0)*(five*c1-four)*(five*c1-three)*(five*c1-one)*c1
     &    +(five/24.d0)*(five*c1-four)*(five*c1-three)*(five*c1-two)*c1
     &    +(one/24.d0)*(five*c1-four)*(five*c1-three)
     &                *(five*c1-two)*(five*c1-one)
        shl(2,1,l)= zero
        shl(3,1,l)= (one/24.d0)*(five*c1-four)*(five*c1-three)
     &                         *(five*c1-two)*(five*c1-one)*c1

        shl(1,2,l)= zero
        shl(2,2,l)=
     &     (five/24.d0)*(five*c2-three)*(five*c2-two)*(five*c2-one)*c2
     &    +(five/24.d0)*(five*c2-four)*(five*c2-two)*(five*c2-one)*c2
     &    +(five/24.d0)*(five*c2-four)*(five*c2-three)*(five*c2-one)*c2
     &    +(five/24.d0)*(five*c2-four)*(five*c2-three)*(five*c2-two)*c2
     &    +(one/24.d0)*(five*c2-four)*(five*c2-three)
     &                *(five*c2-two)*(five*c2-one)
        shl(3,2,l)= (one/24.d0)*(five*c2-four)*(five*c2-three)
     &                         *(five*c2-two)*(five*c2-one)*c2

        shl(1,3,l)=
     &    -(five/24.d0)*(five*c3-three)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-two)*c3
     &    -(one/24.d0)*(five*c3-four)*(five*c3-three)
     &                *(five*c3-two)*(five*c3-one)
        shl(2,3,l)=
     &    -(five/24.d0)*(five*c3-three)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-two)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-one)*c3
     &    -(five/24.d0)*(five*c3-four)*(five*c3-three)*(five*c3-two)*c3
     &    -(one/24.d0)*(five*c3-four)*(five*c3-three)
     &                *(five*c3-two)*(five*c3-one)
        shl(3,3,l)= (one/24.d0)*(five*c3-four)*(five*c3-three)
     &                         *(five*c3-two)*(five*c3-one)*c3

        shl(1,4,l)= (125.d0/24.d0)*(five*c1-two)*(five*c1-one)*c1*c2
     &   + (125.d0/24.d0)*(five*c1-three)*(five*c1-one)*c1*c2
     &   + (125.d0/24.d0)*(five*c1-three)*(five*c1-two)*c1*c2
     &   + (25.d0/24.d0)*(five*c1-three)*(five*c1-two)
     &                  *(five*c1-one)*c2
        shl(2,4,l)= (25.d0/24.d0)*(five*c1-three)
     &                           *(five*c1-two)*(five*c1-one)*c1
        shl(3,4,l)= (25.d0/24.d0)*(five*c1-three)
     &                           *(five*c1-two)*(five*c1-one)*c1*c2

        shl(1,5,l)= (125.d0/12.d0)*(five*c1-one)*c1*(five*c2-one)*c2
     &    +(125.d0/12.d0)*(five*c1-two)*c1*(five*c2-one)*c2
     &    +(25.d0/12.d0)*(five*c1-two)*(five*c1-one)*(five*c2-one)*c2
        shl(2,5,l)= (125.d0/12.d0)*(five*c1-two)*(five*c1-one)*c1*c2
     &    +(25.d0/12.d0)*(five*c1-two)*(five*c1-one)*c1*(five*c2-one)
        shl(3,5,l)= (25.d0/12.d0)*(five*c1-two)
     &                           *(five*c1-one)*c1*(five*c2-one)*c2

        shl(1,6,l)= (125.d0/12.d0)*c1*(five*c2-two)*(five*c2-one)*c2
     &   + (25.d0/12.d0)*(five*c1-one)*(five*c2-two)*(five*c2-one)*c2
        shl(2,6,l)= (125.d0/12.d0)*(five*c1-one)*c1*(five*c2-one)*c2
     &   + (125.d0/12.d0)*(five*c1-one)*c1*(five*c2-two)*c2
     &   + (25.d0/12.d0)*(five*c1-one)*c1*(five*c2-two)*(five*c2-one)
        shl(3,6,l)= (25.d0/12.d0)*(five*c1-one)
     &                           *c1*(five*c2-two)*(five*c2-one)*c2

        shl(1,7,l)= (25.d0/24.d0)*(five*c2-three)
     &                           *(five*c2-two)*(five*c2-one)*c2
        shl(2,7,l)= (125.d0/24.d0)*c1*(five*c2-two)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c1*(five*c2-three)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c1*(five*c2-three)*(five*c2-two)*c2
     &    +(25.d0/24.d0)*c1*(five*c2-three)*(five*c2-two)*(five*c2-one)
        shl(3,7,l)= (25.d0/24.d0)*c1*
     &               (five*c2-three)*(five*c2-two)*(five*c2-one)*c2

        shl(1,8,l)= -(25.d0/24.d0)*(five*c2-three)
     &                            *(five*c2-two)*(five*c2-one)*c2
        shl(2,8,l)=
     &    -(25.d0/24.d0)*(five*c2-three)*(five*c2-two)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c3*(five*c2-two)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c3*(five*c2-three)*(five*c2-one)*c2
     &    +(125.d0/24.d0)*c3*(five*c2-three)*(five*c2-two)*c2
     &    +(25.d0/24.d0)*c3*(five*c2-three)*(five*c2-two)*(five*c2-one)
        shl(3,8,l)= (25.d0/24.d0)*c3*
     &                 (five*c2-three)*(five*c2-two)*(five*c2-one)*c2

        shl(1,9,l)= - (125.d0/12.d0)*c3*(five*c2-two)*(five*c2-one)*c2
     &   - (25.d0/12.d0)*(five*c3-one)*(five*c2-two)*(five*c2-one)*c2
        shl(2,9,l)= -((125.d0/12.d0)*c3)*(five*c2-two)*(five*c2-one)*c2
     &    -((25.d0/12.d0)*(five*c3-one))*(five*c2-two)*(five*c2-one)*c2
     &    +((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-one)*c2
     &    +((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-two)*c2
     &    +((25.d0/12.d0)*(five*c3-one))*c3*(five*c2-two)*(five*c2-one)
        shl(3,9,l)= ((25.d0/12.d0)*(five*c3-one))
     &                     *c3*(five*c2-two)*(five*c2-one)*c2

        shl(1,10,l)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-one)*c2
     &    -((125.d0/12.d0)*(five*c3-two))*c3*(five*c2-one)*c2
     &    -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c2-one)*c2
        shl(2,10,l)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c2-one)*c2
     &    -((125.d0/12.d0)*(five*c3-two))*c3*(five*c2-one)*c2
     &    -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c2-one)*c2
     &    +((125.d0/12.d0)*(five*c3-two))*(five*c3-one)*c3*c2
     &    +((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*c3*(five*c2-one)
        shl(3,10,l)= ((25.d0/12.d0)*(five*c3-two))*(five*c3-one)
     &                      *c3*(five*c2-one)*c2

        shl(1,11,l)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c2
     &   -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c2
     &   -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c2
     &   -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c2
        shl(2,11,l)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c2
     &    -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c2
     &    -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c2
     &  -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c2
     &  +((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c3
        shl(3,11,l)= ((25.d0/24.d0)*(five*c3-three))
     &                     *(five*c3-two)*(five*c3-one)*c3*c2

        shl(1,12,l)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c1
     &    -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c1
     &    -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c1
     &  -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c1
     &  +((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c3
        shl(2,12,l)= -((125.d0/24.d0)*(five*c3-two))*(five*c3-one)*c3*c1
     &    -((125.d0/24.d0)*(five*c3-three))*(five*c3-one)*c3*c1
     &    -((125.d0/24.d0)*(five*c3-three))*(five*c3-two)*c3*c1
     &   -((25.d0/24.d0)*(five*c3-three))*(five*c3-two)*(five*c3-one)*c1
        shl(3,12,l)= ((25.d0/24.d0)*(five*c3-three))
     &                   *(five*c3-two)*(five*c3-one)*c3*c1

        shl(1,13,l)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-one)*c1
     &    -((125.d0/12.d0)*(five*c3 - two))*c3*(five*c1 - one)*c1
     &    -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c1-one)*c1
     &    +((125.d0/12.d0)*(five*c3 - two))*(five*c3 - one)*c3*c1
     &    +((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*c3*(five*c1-one)
        shl(2,13,l)= -((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-one)*c1
     &    -((125.d0/12.d0)*(five*c3-two))*c3*(five*c1-one)*c1
     &    -((25.d0/12.d0)*(five*c3-two))*(five*c3-one)*(five*c1-one)*c1
        shl(3,13,l)= ((25.d0/12.d0)*(five*c3-two))
     &                      *(five*c3-one)*c3*(five*c1-one)*c1

        shl(1,14,l)= -((125.d0/12.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &    -((25.d0/12.d0)*(five*c3-one))*(five*c1-two)*(five*c1-one)*c1
     &    +((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-one)*c1
     &    +((125.d0/12.d0)*(five*c3-one))*c3*(five*c1-two)*c1
     &    +((25.d0/12.d0)*(five*c3-one))*c3*(five*c1-two)*(five*c1-one)
        shl(2,14,l)= -((125.d0/12.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &    -((25.d0/12.d0)*(five*c3-one))*(five*c1-two)*(five*c1-one)*c1
        shl(3,14,l)= ((25.d0/12.d0)*(five*c3-one))
     &                       *c3*(five*c1-two)*(five*c1-one)*c1

        shl(1,15,l)=
     &   -((25.d0/24.d0)*(five*c1-three))*(five*c1-two)*(five*c1-one)*c1
     &   +((125.d0/24.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &   +((125.d0/24.d0)*c3)*(five*c1 - three)*(five*c1 - one)*c1
     &   +((125.d0/24.d0)*c3)*(five*c1 - three)*(five*c1 - two)*c1
     &   +((25.d0/24.d0)*c3)*(five*c1-three)*(five*c1-two)*(five*c1-one)
        shl(2,15,l)= -((25.d0/24.d0)*(five*c1 - three))
     &                       *(five*c1 - two)*(five*c1 - one)*c1
        shl(3,15,l)= (25.d0/24.d0)*c3*(five*c1-three)*(five*c1-two)
     &         *(five*c1-one)*c1


        shl(1,16,l)= ((625.d0/6.d0)*(five*c1-one))*c1*c2*c3
     &    +((625.d0/6.d0)*(five*c1 - two))*c1*c2*c3
     &    +((125.d0/6.d0)*(five*c1 - two))*(five*c1 - one)*c2*c3
     &    -((125.d0/6.d0)*(five*c1 - two))*(five*c1 - one)*c1*c2
        shl(2,16,l)= ((125.d0/6.d0)*c3)*(five*c1-two)*(five*c1-one)*c1
     &    -((125.d0/6.d0)*(five*c1 - two))*(five*c1-one)*c1*c2
        shl(3,16,l)=((125.d0/6.d0)*(five*c1-two))*(five*c1-one)*c1*c2*c3

        shl(1,17,l)= ((625.d0/4.d0)*(five*c2 - one))*c1*c2*c3
     &    +((125.d0/4.d0)*(five*c1 - one))*(five*c2 - one)*c2*c3
     &    -((125.d0/4.d0)*(five*c1 - one))*c1*(five*c2 - one)*c2
        shl(2,17,l)= ((625.d0/4.d0)*(five*c1 - one))*c1*c2*c3
     &    +((125.d0/4.d0)*(five*c1 - one))*(five*c2 - one)*c1*c3
     &    -((125.d0/4.d0)*(five*c1 - one))*c1*(five*c2 - one)*c2
       shl(3,17,l)=((125.d0/4.d0)*(five*c1-one))*(five*c2-one)*c1*c2*c3

        shl(1,18,l)= ((125.d0/6.d0)*c3)*(five*c2-two)*(five*c2 - one)*c2
     &    -((125.d0/6.d0))*c1*(five*c2 - two)*(five*c2 - one)*c2
        shl(2,18,l)= ((625.d0/6.d0)*(five*c2 - one))*c1*c2*c3
     &    +((625.d0/6.d0)*(five*c2 - two))*c1*c2*c3
     &    +((125.d0/6.d0)*(five*c2 - two))*(five*c2 - one)*c1*c3
     &    -((125.d0/6.d0))*c1*(five*c2 - two)*(five*c2 - one)*c2
        shl(3,18,l)=((125.d0/6.d0)*(five*c2-two))*(five*c2-one)*c1*c2*c3

        shl(1,19,l)= -((625.d0/4.d0)*(five*c2 - one))*c1*c2*c3
     &    +((125.d0/4.d0)*(five*c3 - one))*c3*(five*c2 - one)*c2
     &    -((125.d0/4.d0)*(five*c2 - one))*(five*c3 - one)*c1*c2
        shl(2,19,l)= ((625.d0/4.d0)*(five*c3 - one))*c1*c2*c3
     &    -((625.d0/4.d0)*(five*c2 - one))*c1*c2*c3
     &    +((125.d0/4.d0)*(five*c2 - one))*(five*c3 - one)*c1*c3
     &    -((125.d0/4.d0)*(five*c2 - one))*(five*c3 - one)*c1*c2
       shl(3,19,l)= ((125.d0/4.d0)*(five*c2-one))*(five*c3-one)*c1*c2*c3

        shl(1,20,l)= -((625.d0/6.d0)*(five*c3 - one))*c1*c2*c3
     &    -((625.d0/6.d0)*(five*c3 - two))*c1*c2*c3
     &    +((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c3*c2
     &    -((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c1*c2
        shl(2,20,l)= -((625.d0/6.d0)*(five*c3 - one))*c1*c2*c3
     &    -((625.d0/6.d0)*(five*c3 - two))*c1*c2*c3
     &    +((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c3*c1
     &    -((125.d0/6.d0)*(five*c3 - two))*(five*c3 - one)*c1*c2
       shl(3,20,l)= ((125.d0/6.d0)*(five*c3-two))*(five*c3-one)*c1*c2*c3

        shl(1,21,l)= ((625.d0/4.d0)*(five*c3 - one))*c1*c2*c3
     &    -((625.d0/4.d0)*(five*c1 - one))*c1*c2*c3
     &    +((125.d0/4.d0)*(five*c1 - one))*(five*c3 - one)*c2*c3
     &    -((125.d0/4.d0)*(five*c1 - one))*(five*c3 - one)*c1*c2
        shl(2,21,l)= -((625.d0/4.d0)*(five*c1 - one))*c1*c2*c3
     &    +((125.d0/4.d0)*(five*c3 - one))*c3*(five*c1 - one)*c1
     &    -((125.d0/4.d0)*(five*c1 - one))*(five*c3 - one)*c1*c2
       shl(3,21,l)= ((125.d0/4.d0)*(five*c1-one))*(five*c3-one)*c1*c2*c3

            end if

  200      continue
c
           return
           end

c----------------------------------------------------------------------
      subroutine shl2q(shl2,nint,nen)
c----------------------------------------------------------------------
c     program to calculate local second  derivatives
c     for a four-node quadrilateral element
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl2(1,i,l) = local second ("xi") derivative of shape function
c        shl2(2,i,l) = local second ("eta") derivative of shape function
c        shl2(3,i,l) = local second ("xi*eta") derivative of shape function
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension shl2(3,nen,*),ra(16),sa(16)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data r1/0.d00/,
     &     r2/0.577350269189626d00/,
     &     r3a/0.774596669241483d00/,
     &     r3b/0.d00/,
     &     r4a/0.861136311594053d00/,
     &     r4b/0.339981043584856d00/
c
      if (nint.eq.1) then
            ra(1)=r1
            sa(1)=r1
      end if
c
      if(nint.eq.4) then
            ra(1)=r2
            sa(1)=r2
            ra(2)=-r2
            sa(2)=r2
            ra(3)=-r2
            sa(3)=-r2
            ra(4)=r2
            sa(4)=-r2
      end if
c
      if(nint.eq.9) then
            ra(1)=r3a
            sa(1)=r3a
            ra(2)=-r3a
            sa(2)=r3a
            ra(3)=-r3a
            sa(3)=-r3a
            ra(4)=r3a
            sa(4)=-r3a
c
            ra(5)=r3a
            sa(5)=r3b
            ra(6)=r3b
            sa(6)=r3a
            ra(7)=-r3a
            sa(7)=r3b
            ra(8)=r3b
            sa(8)=-r3a
c
            ra(9)=r3b
            sa(9)=r3b
      end if
c
      if(nint.eq.16) then
            ra(1)=r4a
            sa(1)=r4a
            ra(2)=-r4a
            sa(2)=r4a
            ra(3)=-r4a
            sa(3)=-r4a
            ra(4)=r4a
            sa(4)=-r4a
c
            ra(5)=r4b
            sa(5)=r4b
            ra(6)=-r4b
            sa(6)=r4b
            ra(7)=-r4b
            sa(7)=-r4b
            ra(8)=r4b
            sa(8)=-r4b
c
            ra(9)=r4b
            sa(9)=r4a
            ra(10)=-r4b
            sa(10)=r4a
            ra(11)=-r4a
            sa(11)=r4b
            ra(12)=-r4a
            sa(12)=-r4b
            ra(13)=-r4b
            sa(13)=-r4a
            ra(14)=r4b
            sa(14)=-r4a
            ra(15)=r4a
            sa(15)=-r4b
            ra(16)=r4a
            sa(16)=r4b
      end if
c
      do 200 l=1,nint
c
            r=ra(l)
            s=sa(l)
            shl2(1,1,l)=zero
            shl2(2,1,l)=zero
            shl2(3,1,l)=pt25
            shl2(1,2,l)=zero
            shl2(2,2,l)=zero
            shl2(3,2,l)=-pt25
            shl2(1,3,l)=zero
            shl2(2,3,l)=zero
            shl2(3,3,l)=pt25
            shl2(1,4,l)=zero
            shl2(2,4,l)=zero
            shl2(3,4,l)=-pt25
            if(nen.eq.9) then
                  onepr=one+r
                  onemr=one-r
                  oneps=one+s
                  onems=one-s
		  onemrs=one-r*r
		  onemss=one-s*s
                  shl2(1,5,l)=-oneps
                  shl2(2,5,l)=zero
                  shl2(3,5,l)=-r
                  shl2(1,6,l)=zero
                  shl2(2,6,l)=-onemr
                  shl2(3,6,l)=s
                  shl2(1,7,l)=-onems
                  shl2(2,7,l)=zero
                  shl2(3,7,l)=r
                  shl2(1,8,l)=zero
                  shl2(2,8,l)=-onepr
                  shl2(3,8,l)=-s
                  shl2(1,9,l)=-two*onemss
                  shl2(2,9,l)=-two*onemrs
                  shl2(3,9,l)=four*r*s
c
                  do 1111 k=5,8
                        do 2222 i=1,3
                              shl2(i,k,l)=shl2(i,k,l)-pt5*shl2(i,9,l)
2222                    continue
1111              continue
c
                  do 3333 i=1,3
                        shl2(i,1,l)=shl2(i,1,l)
     &                  -pt5*(shl2(i,5,l)+shl2(i,8,l))-pt25*shl2(i,9,l)
                        shl2(i,2,l)=shl2(i,2,l)
     &                  -pt5*(shl2(i,6,l)+shl2(i,5,l))-pt25*shl2(i,9,l)
                        shl2(i,3,l)=shl2(i,3,l)
     &                  -pt5*(shl2(i,7,l)+shl2(i,6,l))-pt25*shl2(i,9,l)
                        shl2(i,4,l)=shl2(i,4,l)
     &                  -pt5*(shl2(i,8,l)+shl2(i,7,l))-pt25*shl2(i,9,l)
3333              continue
            end if
            if (nen.eq.16) then
                  onemrsq=one-r*r
                  onemssq=one-s*s
                  onep3r=one+three*r
                  onem3r=one-three*r
                  onep3s=one+three*s
                  onem3s=one-three*s
c
c	not inplemented
c
c
            end if
  200 continue
c
      return
      end

c----------------------------------------------------------------------
      subroutine shgqs(xl,det,shl,shg,nint,nel,neg,quad,nen,
     &     shlnode,nenode)
c----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions and
c        jacobian determinants for a  quadrilateral element
c
c        xl(j,i)    = global coordinates
c        det(l)     = jacobian determinant
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c        shg(1,i,l) = x-derivative of shape function
c        shg(2,i,l) = y-derivative of shape function
c        shg(3,i,l) = shl(3,i,l)
c        xs(i,j)    = jacobian matrix
c                 i = local node number or global coordinate number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points, eq. 1 or 4
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical quad
      dimension xl(2,*),det(*),xs(2,2)
      dimension shl(3,nen,*),shg(3,nen,*),shlnode(3,nenode,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      call move(shg,shl,3*nen*nint)
c
      do l=1,nint
c
         if (.not.quad) then
            do i=1,3
               shg(i,3,l) = shl(i,3,l) + shl(i,4,l)
               shg(i,4,l) = zero
            end do
         endif
c
         do j=1,2
            do i=1,2
               xs(i,j) = rowdot(shlnode(i,1,l),xl(j,1),3,2,nenode)
            end do
         end do
c
         det(l) = xs(1,1)*xs(2,2)-xs(1,2)*xs(2,1)
c
         if (det(l).le.zero) then
            write(ieco,1000) nel,neg
            stop
         endif
c
         do j=1,2
            do i=1,2
               xs(i,j) = xs(i,j)/det(l)
            end do
         end do
c
         do i=1,nen
            temp = xs(2,2)*shg(1,i,l) - xs(1,2)*shg(2,i,l)
            shg(2,i,l) = - xs(2,1)*shg(1,i,l) + xs(1,1)*shg(2,i,l)
            shg(1,i,l) = temp
         end do
c
      end do
      return
c
 1000 format(///,'shgqs - non-positive determinant - element ',i10,
     &     ' in element group  ',i10,'  shgqs')
      end

c----------------------------------------------------------------------
      subroutine shgtqsd(ien,xl,shl,shg,nside,nint,nints,nel,neg,nen,
     &                   shlnode,nenode)
c----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions and
c        jacobian determinants for a  quadrilateral element
c        xl(j,i)    = global coordinates
c        det(l)     = jacobian determinant
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c        shg(1,i,l) = x-derivative of shape function
c        shg(2,i,l) = y-derivative of shape function
c        shg(3,i,l) = shl(3,i,l)
c        xs(i,j)    = jacobian matrix
c                 i = local node number or global coordinate number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points, eq. 1 or 4
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension xl(2,*),shl(3,nen,*),shg(3,nen,*),xs(2,2),
     &     shlnode(3,nenode,*),ien(*)
      dimension igas(100)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite

ccc      call move(shg,shl,3*nen*nint)

c
c     renumeracao
c
      ns = 1
      nl1=ien(1)
      nl2=ien(2)
c
      if(nl2.gt.nl1) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
      else
         do  nn=1,nints
            ng = (ns-1)*nints + nn
            ngs = (ns-1)*nints
            igas(ng) = ngs + nints + 1 - nn
         end do
      end if
c
c
      ns = 2
      nl1=ien(2)
      nl2=ien(3)
c
      if(nl2.gt.nl1) then
         do  nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
      else
         do  nn=1,nints
            ng = (ns-1)*nints + nn
            ngs = (ns-1)*nints
            igas(ng) = ngs + nints + 1 - nn
         end do
      end if
c
c
      if(nside.eq.3) then
         ns = 3
         nl1=ien(3)
         nl2=ien(1)
c
         if(nl2.gt.nl1) then
            do  nn=1,nints
               ng = (ns-1)*nints + nn
               igas(ng) = ng
	    end do
         else
            do  nn=1,nints
               ng = (ns-1)*nints + nn
               ngs = (ns-1)*nints
	       igas(ng) = ngs + nints + 1 - nn
            end do
         end if
c
      end if
c
      if(nside.eq.4) then
         ns = 3
         nl1=ien(3)
         nl2=ien(4)
c
         if(nl2.gt.nl1) then
            do  nn=1,nints
               ng = (ns-1)*nints + nn
               igas(ng) = ng
	    end do
         else
            do  nn=1,nints
               ng = (ns-1)*nints + nn
               ngs = (ns-1)*nints
               igas(ng) = ngs + nints + 1 - nn
	    end do
         end if
c
         ns = 4
         nl1=ien(4)
         nl2=ien(1)
c
         if(nl2.gt.nl1) then
            do  nn=1,nints
               ng = (ns-1)*nints + nn
               igas(ng) = ng
	    end do
         else
            do  nn=1,nints
               ng = (ns-1)*nints + nn
               ngs = (ns-1)*nints
               igas(ng) = ngs + nints + 1 - nn
	    end do
         end if
c
      end if
c
c     integration points
c
      do l=1,nint
c
         do j=1,2
            do i=1,2
               xs(i,j) = rowdot(shlnode(i,1,l),xl(j,1),3,2,nenode)
            end do
         end do
c
         deta = xs(1,1)*xs(2,2)-xs(1,2)*xs(2,1)
         if (deta.le.zero) then
            write(iecho,1000) nel,neg
            stop
         endif
c
         do j=1,2
            do i=1,2
               xs(i,j) = xs(i,j)/deta
            end do
         end do
c
         do i=1,nen
            lga = igas(l)
            shg(3,i,lga) = shl(3,i,l)
            temp = xs(2,2)*shl(1,i,l) - xs(1,2)*shl(2,i,l)
            shg(2,i,lga) = - xs(2,1)*shl(1,i,l) + xs(1,1)*shl(2,i,l)
            shg(1,i,lga) = temp
         end do
c
      end do
c
      return
c
 1000 format(///,'shgtqsd - non-positive determinant - element ',i10,
     &     ' in element group  ',i10,'  shgtqsd')
      end

c-----------------------------------------------------------------------
      subroutine shghx(xl,det,shl,shg,nint,nel,neg,hexa,nen,
     &                 shlnode,nenode)
c-----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions and
c     jacobian determinants for a HEXAHEDRAL element
c        xl(j,i)    = global coordinates
c        det(l)     = jacobian determinant
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local ("zeta") derivative of shape function
c        shl(4,i,l) = local  shape function
c
c        shg(1,i,l) = x-derivative of shape function
c        shg(2,i,l) = y-derivative of shape function
c        shg(3,i,l) = z-derivative of shape function
c        shg(4,i,l) = shl(4,i,l)
c
c           xs(i,j) = jacobian matrix
c         jinv(i,j) = inverse jacobian matrix
c                 i = local node number or global coordinate number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical hexa
      dimension xl(3,*),det(*),xjac(3,3),xjinv(3,3)
      dimension shl(4,nen,*),shg(4,nen,*),shlnode(4,nenode,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      call move(shg,shl,4*nen*nint)

c      write(*,*) "DADOS HEX -> shlnode em SHGHX funcao"
c      do ii=1,nint
c         write(*,999) (shlnode(4,in,ii),in=1,nen)
c      end do
c      write(*,*) "DADOS HEX -> shlnode em SHGHX deriv 1"
c      do ii=1,nint
c         write(*,999) (shlnode(1,in,ii),in=1,nen)
c      end do
c      write(*,*) "DADOS HEX -> shlnode em SHGHX deriv 2"
c      do ii=1,nint
c         write(*,999) (shlnode(2,in,ii),in=1,nen)
c      end do
c      write(*,*) "DADOS HEX -> shlnode em SHGHX deriv 3"
c      do ii=1,nint
c         write(*,999) (shlnode(3,in,ii),in=1,nen)
c      end do
c
c     loop pontos integracao - calcula isoparametrico
c
      do l=1,nint
c
         do j=1,3
            do i=1,3
               xjac(i,j) = rowdot(shlnode(i,1,l),xl(j,1),4,3,nenode)
            end do
         end do
c
         det(l) = xjac(1,1)*(xjac(2,2)*xjac(3,3) - xjac(3,2)*xjac(2,3))
     &          + xjac(1,2)*(xjac(3,1)*xjac(2,3) - xjac(2,1)*xjac(3,3))
     &          + xjac(1,3)*(xjac(2,1)*xjac(3,2) - xjac(3,1)*xjac(2,2))
c
c         write(*,*) "det(l)=",det(l)
c
         if (det(l).le.zero) then
            write(ieco,1000) nel,neg
            write(*,1000) nel,neg
            stop
         endif
c
c     invert jacobian xjac and store its inverse in jinv
c
         detinv = 1.0/det(l)
         xjinv(1,1) = +detinv*(xjac(2,2)*xjac(3,3)-xjac(2,3)*xjac(3,2))
         xjinv(2,1) = -detinv*(xjac(2,1)*xjac(3,3)-xjac(2,3)*xjac(3,1))
         xjinv(3,1) = +detinv*(xjac(2,1)*xjac(3,2)-xjac(2,2)*xjac(3,1))
         xjinv(1,2) = -detinv*(xjac(1,2)*xjac(3,3)-xjac(1,3)*xjac(3,2))
         xjinv(2,2) = +detinv*(xjac(1,1)*xjac(3,3)-xjac(1,3)*xjac(3,1))
         xjinv(3,2) = -detinv*(xjac(1,1)*xjac(3,2)-xjac(1,2)*xjac(3,1))
         xjinv(1,3) = +detinv*(xjac(1,2)*xjac(2,3)-xjac(1,3)*xjac(2,2))
         xjinv(2,3) = -detinv*(xjac(1,1)*xjac(2,3)-xjac(1,3)*xjac(2,1))
         xjinv(3,3) = +detinv*(xjac(1,1)*xjac(2,2)-xjac(1,2)*xjac(2,1))
c
c         write(*,*) "NOVO INVERSA DO JACOBIANO"
c         do i=1,3
c            write(*,*) (xjinv(i,j),j=1,3)
c         end do

c
c     calcula derivadas "globais" com rel. x y z
c
         do i=1,nen
            shg(1,i,l) = xjinv(1,1)*shl(1,i,l)
     &                 + xjinv(1,2)*shl(2,i,l)
     &                 + xjinv(1,3)*shl(3,i,l)

            shg(2,i,l) = xjinv(2,1)*shl(1,i,l)
     &                 + xjinv(2,2)*shl(2,i,l)
     &                 + xjinv(2,3)*shl(3,i,l)

            shg(3,i,l) = xjinv(3,1)*shl(1,i,l)
     &                 + xjinv(3,2)*shl(2,i,l)
     &                 + xjinv(3,3)*shl(3,i,l)

            shg(4,i,l) = shl(4,i,l)
         end do
c
      end do

c      write(*,*) "DADOS HEX -> SHGHX FINAL - DERIV 1-2-3 FUNCAO"
c      do ii=1,nint
c         write(*,*) "PONTO INTEGRACAO ",ii
c         do in=1,nen
c            write(*,'(9F8.4)') shg(1,in,ii),shg(2,in,ii),
c     &                   shg(3,in,ii),shg(4,in,ii)
c         end do
c      end do
c
      return
c
 1000 format(///,'shghx - non-positive determinant - element ',i10,
     &          ' in element group  ',i10,'  shghx')
      end

c----------------------------------------------------------------------
      subroutine shgthxsd(ien,xl,shl,shg,nside,nint,nints,
     &                    nel,neg,nen,shlnode,nenode)
c----------------------------------------------------------------------
c     program to calculate global derivatives of shape functions and
c     jacobian determinants for a  quadrilateral element
c        xl(j,i)    = global coordinates
c        det(l)     = jacobian determinant
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local ("zeta") derivative of shape function
c        shl(4,i,l) = local  shape function
c
c        shg(1,i,l) = x-derivative of shape function
c        shg(2,i,l) = y-derivative of shape function
c        shg(3,i,l) = z-derivative of shape function
c        shg(4,i,l) = shl(4,i,l)
c
c        xs(i,j)    = jacobian matrix
c                 i = local node number or global coordinate number
c                 j = global coordinate number
c                 l = integration-point number
c              nint = number of integration points, eq. 1 or 4
c
c     based on shgtqsd
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension xl(3,*),ien(*)
      dimension shl(4,nen,*),shg(4,nen,*),shlnode(4,nenode,*)
      dimension xs(3,3),xsinv(3,3)
      dimension co(3),vab(3),vad(3),xn(3),vc(3)
      dimension igas(100)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite

ccc   call move(shg,shl,4*nen*nint)

c
c     calcula centroide do elemento
c
      co(1) = 0.d0
      co(2) = 0.d0
      co(3) = 0.d0
      do i=1,8
         co(1) = co(1) + xl(1,i)
         co(2) = co(2) + xl(2,i)
         co(3) = co(3) + xl(3,i)
      end do
      co(1)=co(1)/8.d0
      co(2)=co(2)/8.d0
      co(3)=co(3)/8.d0

c     DEBUG
      write(*,'(A)') "shgthxsd"
      write(*,'(A,2I5)') "nintb, nside ",nint, nside
      if(nside.ne.6) then
         write(*,'(A,I5)') "ERROR: nside in shgthxsd must be 6"
         stop
      end if
c
c     renumeracao
c     face 1
c
      ns=1
      nl1=ien(1)
      nl2=ien(2)
      nl3=ien(6)
      nl4=ien(5)
      call vec3sub(xl(1,2), xl(1,1), vab)
      call vec3sub(xl(1,5), xl(1,1), vad)
      call vec3sub(co,xl(1,1),vc)
      call cross(vab,vad,xn)
      call nrm3(xn)
      dotcn = dot3(vc,xn)
      write(*,'(4I5,4F10.4)') nl1,nl2,nl3,nl4,xn(1),xn(2),xn(3),dotcn
      if(dotcn.lt.0.d0) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
c$$$      else
c$$$         do  nn=1,nints
c$$$            ng = (ns-1)*nints + nn
c$$$            ngs = (ns-1)*nints
c$$$            igas(ng) = ngs + nints + 1 - nn
c$$$         end do
      end if
c
c     face 2
c
      ns=2
      nl1=ien(1)
      nl2=ien(4)
      nl3=ien(8)
      nl4=ien(5)
      call vec3sub(xl(1,4), xl(1,1), vab)
      call vec3sub(xl(1,8), xl(1,1), vad)
      call vec3sub(co,xl(1,1),vc)
      call cross(vab,vad,xn)
      call nrm3(xn)
      dotcn = dot3(vc,xn)
      write(*,'(4I5,4F10.4)') nl1,nl2,nl3,nl4,xn(1),xn(2),xn(3),dotcn
      if(dotcn.lt.0.d0) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
c$$$      else
c$$$         do  nn=1,nints
c$$$            ng = (ns-1)*nints + nn
c$$$            ngs = (ns-1)*nints
c$$$            igas(ng) = ngs + nints + 1 - nn
c$$$         end do
      end if
c
c     face 3
c
      ns=3
      nl1=ien(2)
      nl2=ien(3)
      nl3=ien(7)
      nl4=ien(6)
      call vec3sub(xl(1,3), xl(1,2), vab)
      call vec3sub(xl(1,7), xl(1,2), vad)
      call vec3sub(co,xl(1,2),vc)
      call cross(vab,vad,xn)
      call nrm3(xn)
      dotcn = dot3(vc,xn)
      write(*,'(4I5,4F10.4)') nl1,nl2,nl3,nl4,xn(1),xn(2),xn(3),dotcn
      if(dotcn.lt.0.d0) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
c$$$      else
c$$$         do  nn=1,nints
c$$$            ng = (ns-1)*nints + nn
c$$$            ngs = (ns-1)*nints
c$$$            igas(ng) = ngs + nints + 1 - nn
c$$$         end do
      end if
c
c     face 4
c
      ns=4
      nl1=ien(4)
      nl2=ien(3)
      nl3=ien(7)
      nl4=ien(8)
      call vec3sub(xl(1,3), xl(1,4), vab)
      call vec3sub(xl(1,7), xl(1,4), vad)
      call vec3sub(co,xl(1,4),vc)
      call cross(vab,vad,xn)
      call nrm3(xn)
      dotcn = dot3(vc,xn)
      write(*,'(4I5,4F10.4)') nl1,nl2,nl3,nl4,xn(1),xn(2),xn(3),dotcn
      if(dotcn.lt.0.d0) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
c$$$      else
c$$$         do  nn=1,nints
c$$$            ng = (ns-1)*nints + nn
c$$$            ngs = (ns-1)*nints
c$$$            igas(ng) = ngs + nints + 1 - nn
c$$$         end do
      end if
c
c     face 5
c
      ns=5
      nl1=ien(1)
      nl2=ien(2)
      nl3=ien(3)
      nl4=ien(4)
      call vec3sub(xl(1,2), xl(1,1), vab)
      call vec3sub(xl(1,3), xl(1,1), vad)
      call vec3sub(co,xl(1,1),vc)
      call cross(vab,vad,xn)
      call nrm3(xn)
      dotcn = dot3(vc,xn)
      write(*,'(4I5,4F10.4)') nl1,nl2,nl3,nl4,xn(1),xn(2),xn(3),dotcn
      if(dotcn.lt.0.d0) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
c$$$      else
c$$$         do  nn=1,nints
c$$$            ng = (ns-1)*nints + nn
c$$$            ngs = (ns-1)*nints
c$$$            igas(ng) = ngs + nints + 1 - nn
c$$$         end do
      end if
c
c     face 6
c
      ns=6
      nl1=ien(5)
      nl2=ien(6)
      nl3=ien(7)
      nl4=ien(8)
      call vec3sub(xl(1,6), xl(1,5), vab)
      call vec3sub(xl(1,7), xl(1,5), vad)
      call vec3sub(co,xl(1,5),vc)
      call cross(vab,vad,xn)
      call nrm3(xn)
      dotcn = dot3(vc,xn)
      write(*,'(4I5,4F10.4)') nl1,nl2,nl3,nl4,xn(1),xn(2),xn(3),dotcn
      if(dotcn.lt.0.d0) then
         do nn=1,nints
            ng = (ns-1)*nints + nn
            igas(ng) = ng
         end do
c$$$      else
c$$$         do nn=1,nints
c$$$            ng = (ns-1)*nints + nn
c$$$            ngs = (ns-1)*nints
c$$$            igas(ng) = ngs + nints + 1 - nn
c$$$         end do
      end if
c
c     integration points
c
      do l=1,nint
c
         do j=1,3
            do i=1,3
               xs(i,j) = rowdot(shlnode(i,1,l),xl(j,1),4,3,nenode)
            end do
         end do
c
         deta = xs(1,1)*(xs(2,2)*xs(3,3) - xs(3,2)*xs(2,3))
     &        + xs(1,2)*(xs(3,1)*xs(2,3) - xs(2,1)*xs(3,3))
     &        + xs(1,3)*(xs(2,1)*xs(3,2) - xs(3,1)*xs(2,2))
c
         if (deta.le.zero) then
            write(iecho,1000) nel,neg
            stop
         endif
c
         detinv = 1.0/deta
         xsinv(1,1) = +detinv * (xs(2,2)*xs(3,3) - xs(2,3)*xs(3,2))
         xsinv(2,1) = -detinv * (xs(2,1)*xs(3,3) - xs(2,3)*xs(3,1))
         xsinv(3,1) = +detinv * (xs(2,1)*xs(3,2) - xs(2,2)*xs(3,1))
         xsinv(1,2) = -detinv * (xs(1,2)*xs(3,3) - xs(1,3)*xs(3,2))
         xsinv(2,2) = +detinv * (xs(1,1)*xs(3,3) - xs(1,3)*xs(3,1))
         xsinv(3,2) = -detinv * (xs(1,1)*xs(3,2) - xs(1,2)*xs(3,1))
         xsinv(1,3) = +detinv * (xs(1,2)*xs(2,3) - xs(1,3)*xs(2,2))
         xsinv(2,3) = -detinv * (xs(1,1)*xs(2,3) - xs(1,3)*xs(2,1))
         xsinv(3,3) = +detinv * (xs(1,1)*xs(2,2) - xs(1,2)*xs(2,1))
c
         do i=1,nen
c
c            lga=igas(l)
c
            shg(1,i,l) = xsinv(1,1)*shl(1,i,l)
     &                   + xsinv(1,2)*shl(2,i,l)
     &                   + xsinv(1,3)*shl(3,i,l)

            shg(2,i,l) = xsinv(2,1)*shl(1,i,l)
     &                   + xsinv(2,2)*shl(2,i,l)
     &                   + xsinv(2,3)*shl(3,i,l)

            shg(3,i,l) = xsinv(3,1)*shl(1,i,l)
     &                   + xsinv(3,2)*shl(2,i,l)
     &                   + xsinv(3,3)*shl(3,i,l)

            shg(4,i,l) = shl(4,i,l)
         end do
      end do
c
      return
c
 1000 format(///,'shgthxsd - non-positive determinant - element ',i10,
     &     ' in element group  ',i10,'  shgthxsd')
      end

c-----------------------------------------------------------------------
      subroutine elmlib(ntype,mpnpar,itask,neg)
c-----------------------------------------------------------------------
c     program to call element routines
c-----------------------------------------------------------------------
      common a(1)
c
      go to (10,20) ntype
c
c     Poisson problem - kinematic formulation
c
  10  continue
      call pflux(itask,a(mpnpar),a(mpnpar+16),neg)
 20   continue
      return
      end

c-----------------------------------------------------------------------
      subroutine pflux(itask,npar,mp,neg)
c-----------------------------------------------------------------------
c     program to set storage and call tasks for the
c     primal mixed Poisson  problem
c     with continuous temperature and discontinuous flux
c-----------------------------------------------------------------------
      dimension npar(*),mp(*)
      common /bpoint/ mfirst,mlast,ilast,mtot,iprec
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /info  / iexec,iprtin,irank,nsd,numnp,ndof,nlvect,
     &                numeg,nmultp,nedge

      common /spoint/ mpd,mpx,mpid,mpf,mpdiag,mpngrp,
     &                mpalhs,mpdlhs,mpbrhs,mped,index

      character*4 ia
      common a(1)
      common /dictn/ ia(10000000)
c
      write(*,*)
      write(*,'(A)') "subroutine pflux"
c
      mw     = 1
      mdet   = 2
      mshl   = 3
      mshg   = 4
c
      mc     = 5
      mgrav  = 6
      mien   = 7
      mmat   = 8
      mlm    = 9
c
      mxl    = 10
      mdl    = 11
c
c     pointers for condensation
c
      mipar  = 12
      mlado  = 13
c
c
      mdetc  = 14
      mshlc  = 15
      mshgc  = 16
      mwc    = 17
c
      melefd = 18
      melred = 19
      mdlf   = 20
      mdlp   = 21
      mdsfl  = 22
c
c    inetrais nas arestas
c
      mdetpn = 23
      mshlpn = 24
      mshgpn = 25
      mwpn   = 26
c
c     pointers for boundary integrals
c
      mdside = 27
      mxls   = 28
      midlsd = 29
c
c     geometria das arestas
c
      mdetn  = 30
      mshln  = 31
      mshgn  = 32
      mwn    = 33
c
c    valores na fronteira
c
      mdetb  = 34
      mshlb  = 35
      mshgb  = 36
      mddis  = 37
c
      mdetp  = 38
      mshlp  = 39
      mshgp  = 40
      mwp    = 41
c
c    matrizes e vetores da formulacao hibrida
c
      melma   = 42
      melmb   = 43
      melmc   = 44
      melmd   = 45
      melmh   = 46

      melmbb  = 47
      melmcb  = 48
      melmhb  = 49

      melfa   = 50
      melfb   = 51
      melfc   = 52
      melfd   = 53
      melfab  = 54
c
      melfbb  = 55
      melfcb  = 56
c
      mshsde  = 57
c
      mshlpsd = 58
      mshgpsd = 59
      mshlcsd = 60
      mshgcsd = 61
c
      melmdb  = 62
      melmbc  = 63
c
c     parametros
c
      ntype  = npar( 1)
      numel  = npar( 2)
      numat  = npar( 3)
      nint   = npar( 4)
      nen    = npar( 5)
      nencon = npar( 6)
      nenp   = npar( 7)
      npars  = npar( 8)
      nints  = npar( 9)
      nface  = npar(10)
c
c     2D elements
c
      if(nsd.eq.2) then
c
         write(*,'(A,I5)') " elementos 2D"
         if(nen.eq.0) nen=4
c
c     quad
c
        if(nen.eq.4)  nenlad = 2
        if(nen.eq.8)  nenlad = 3
        if(nen.eq.9)  nenlad = 3
        if(nen.eq.16) nenlad = 4
c
c     triangle
c
        if(nen.eq.3)  nenlad = 2
        if(nen.eq.6)  nenlad = 3
        if(nen.eq.10) nenlad = 4
c
        if(nencon.eq.4) then
           nnods = 2
           nside = 4
        end if
c
        if(nencon.eq.8.or.nencon.eq.9) then
           nnods = 3
           nside = 4
        end if
c
        if(nencon.eq.16) then
           nnods = 4
           nside = 4
        end if
c
        if(nencon.eq.25) then
           nnods = 5
           nside = 4
        end if
c
        if(nencon.eq.36) then
           nnods = 6
           nside = 4
        end if
c
        if(nencon.eq.49) then
           nnods = 7
           nside = 4
        end if
c
        if(nencon.eq.64) then
           nnods = 8
           nside = 4
        end if
c
        if(nencon.eq.3) then
           nnods  = 2
           nside  = 3
        end if
c
        if(nencon.eq.6) then
           nnods  = 3
           nside  = 3
        end if
c
        nintb=nside*nints
c
      end if
c
c     3D elements
c     TODO: conferir se as configuracoes estao certas...
c
      if (nsd.eq.3) then
c
         write(*,'(A)') " elementos 3D"
c
         if(nen.eq.8)  nenlad = 4
c
         if(nencon.eq.8) then
            nnods = 4
            nside = 6
         end if
c
         nintb=nside*nints
c
      end if
c
c     parameters for post processing
c
      nodsp=npars*nside
c
c     set element parameters
c
      ndimc  = 10
      ned    = 1
      ncon   = 2
      nee    = nodsp*ned
      neep   = nenp*ned
      necon  = nencon*ncon
      neesq  = nee*nee
      ngrav  = 3
c
      if(nsd.eq.2) then
         nesd    = 2
         nrowsh  = 3
         nrowsh3 = 2
      else if(nsd.eq.3) then
         nesd    = 3
         nrowsh  = 3
         nrowsh3 = 4
      end if
c
c     *** DEBUG ***
c
      write(*,'(A,I5)') " nnods   ", nnods
      write(*,'(A,I5)') " nside   ", nside
      write(*,'(A,I5)') " ntype   ", ntype
      write(*,'(A,I5)') " numel   ", numel
      write(*,'(A,I5)') " numat   ", numat
      write(*,'(A,I5)') " nen     ", nen
      write(*,'(A,I5)') " nencon  ", nencon
      write(*,'(A,I5)') " nenp    ", nenp
      write(*,'(A,I5)') " npars   ", npars
      write(*,'(A,I5)') " nint    ", nint
      write(*,'(A,I5)') " nints   ", nints
      write(*,'(A,I5)') " nintb   ", nintb
      write(*,'(A,I5)') " nface   ", nface
      write(*,'(A,I5)') " neep    ", neep
      write(*,'(A,I5)') " nrowsh  ", nrowsh
      write(*,'(A,I5)') " nrowsh3 ", nrowsh3
      write(*,'(A,I5)') " nsd     ",nsd
      write(*,'(A,I5)') " nedge   ",nedge
      write(*,'(A,I5)') " numnp   ",numnp
      write(*,'(A,I5)') " numel   ",numel
      write(*,'(A,I8)') " nmultp  ",nmultp
      write(*,'(A,I5)')
c
c     *** DEBUG ***
c
      if (itask.eq.1) then
c
c     set memory pointers
c        note:  the mp array is stored directly after the npar array,
c               beginning at location mpnpar + 16 of blank common.
c               the variable "junk" is not used subsequently.
c
         junk       = mpoint('mp      ',80     ,0     ,0     ,1)
c
         mp(mw    ) = mpoint('w       ',nint   ,0     ,0     ,iprec)
         mp(mdet  ) = mpoint('det     ',nint   ,0     ,0     ,iprec)
         mp(mshl  ) = mpoint('shl     ',nrowsh3,nen   ,nint  ,iprec)
         mp(mshg  ) = mpoint('shg     ',nrowsh3,nen   ,nint  ,iprec)
         mp(mc    ) = mpoint('c       ',ndimc  ,numat ,0     ,iprec)
         mp(mgrav ) = mpoint('grav    ',ngrav  ,0     ,0     ,iprec)
cc
         mp(mien  ) = mpoint('ien     ',nen    ,numel ,0     ,1)
         mp(mmat  ) = mpoint('mat     ',numel  ,0     ,0     ,1)
cc
         mp(mlm   ) = mpoint('lm      ',ned    ,nodsp ,numel ,1)
         mp(mxl   ) = mpoint('xl      ',nesd   ,nen   ,0     ,iprec)
         mp(mdl   ) = mpoint('dl      ',ned    ,nodsp ,0     ,iprec)
c
         mp(mipar ) = mpoint('ipar    ',nodsp  ,numel ,0,     1)
         mp(mlado)  = mpoint('lado    ',nside  ,numel ,0     ,1)
c
         mp(mwc   ) = mpoint('wc      ',nint   ,0     ,0     ,iprec)
         mp(mdetc ) = mpoint('detc    ',nint   ,0     ,0     ,iprec)
         mp(mshlc ) = mpoint('shlc    ',nrowsh3,nencon,nint  ,iprec)
         mp(mshgc ) = mpoint('shgc    ',nrowsh3,nencon,nint  ,iprec)
         mp(melefd) = mpoint('eleffd  ',nee    ,nee   ,0     ,iprec)
         mp(melred) = mpoint('elresd  ',nee    ,0     ,0     ,iprec)
         mp(mdlf  ) = mpoint('dlf     ',ncon   ,nencon,0     ,iprec)
         mp(mdlp  ) = mpoint('dlp     ',ned    ,nenp  ,0     ,iprec)
         mp(mdsfl ) = mpoint('dsfl    ',ncon   ,nencon,numel ,iprec)
c
         mp(mdetpn) = mpoint('detpn   ',nints  ,0     ,0     ,iprec)
         mp(mshlpn) = mpoint('shlpn   ',nrowsh ,npars ,nints ,iprec)
         mp(mshgpn) = mpoint('shgpn   ',nrowsh ,npars ,nints ,iprec)
         mp(mwpn  ) = mpoint('wpn     ',nints  ,0     ,0     ,iprec)
c
         mp(mdside) = mpoint('idside  ',nside  ,nnods ,0     ,1    )
         mp(mxls  ) = mpoint('xls     ',nesd   ,nenlad,0     ,iprec)
         mp(midlsd) = mpoint('idlsd   ',nnods  ,0     ,0     ,1    )
         mp(mdetn ) = mpoint('detn    ',nints  ,0     ,0     ,iprec)
         mp(mshln ) = mpoint('shln    ',nrowsh ,nnods ,nints ,iprec)
         mp(mshgn ) = mpoint('shgn    ',nrowsh ,nnods ,nints ,iprec)
         mp(mwn   ) = mpoint('wn      ',nints  ,0     ,0     ,iprec)
c
         mp(mdetb ) = mpoint('detb    ',nints  ,0     ,0     ,iprec)
         mp(mshlb ) = mpoint('shlb    ',nrowsh ,nenlad,nints ,iprec)
         mp(mshgb ) = mpoint('shgb    ',nrowsh ,nenlad,nints ,iprec)
c
         mp(mddis ) = mpoint('ddis    ',ned    ,nenp  ,numel ,iprec )
         mp(mwp   ) = mpoint('wp      ',nint   ,0     ,0     ,iprec)
         mp(mdetp ) = mpoint('detp    ',nint   ,0     ,0     ,iprec)
         mp(mshlp ) = mpoint('shlp    ',nrowsh3,nenp  ,nint  ,iprec)
         mp(mshgp ) = mpoint('shgp    ',nrowsh3,nenp  ,nint  ,iprec)
c
         mp(melma ) = mpoint('elma    ',necon  ,necon ,0     ,iprec)
         mp(melmb ) = mpoint('elmb    ',necon  ,neep  ,0     ,iprec)
         mp(melmc ) = mpoint('elmc    ',necon  ,nee   ,0     ,iprec)
         mp(melmd ) = mpoint('elmd    ',neep   ,necon ,0     ,iprec)
         mp(melmh ) = mpoint('elmh    ',neep   ,neep  ,0     ,iprec)
c
         mp(melmbb) = mpoint('elmbb   ',neep   ,neep  ,0     ,iprec)
         mp(melmcb) = mpoint('elmcb   ',neep   ,nee   ,0     ,iprec)
         mp(melmhb) = mpoint('elmhb   ',nee    ,necon ,0     ,iprec)
c
         mp(melfa ) = mpoint('elfa    ',necon  ,0     ,0     ,iprec)
         mp(melfb ) = mpoint('elfb    ',neep   ,0     ,0     ,iprec)
         mp(melfc ) = mpoint('elfc    ',nee    ,0     ,0     ,iprec)
         mp(melfd ) = mpoint('elfd    ',necon  ,0     ,0     ,iprec)
c
         mp(melfab) = mpoint('elfab   ',necon  ,0     ,0     ,iprec)
         mp(melfbb) = mpoint('elfbb   ',neep   ,0     ,0     ,iprec)
         mp(melfcb) = mpoint('elfcb   ',nee    ,0     ,0     ,iprec)
c
         mp(mshsde) = mpoint('shsde   ',nrowsh3, nen   ,nintb ,iprec)
c
         mp(mshlpsd) = mpoint('shlpsd  ',nrowsh3, nenp  ,nintb ,iprec)
         mp(mshgpsd) = mpoint('shgpsd  ',nrowsh3, nenp  ,nintb ,iprec)
         mp(mshlcsd) = mpoint('shlcsd  ',nrowsh3, nencon,nintb ,iprec)
         mp(mshgcsd) = mpoint('shgcsd  ',nrowsh3, nencon,nintb ,iprec)
c
         mp(melmdb) = mpoint('elmdb   ',nee    ,neep   ,0     ,iprec)
         mp(melmbc) = mpoint('elmbc   ',nee    ,neep   ,0     ,iprec)
c
      endif
c
c     task calls
c
      if (itask.gt.4) return
      go to (100,200,300,400),itask
c
  100 continue
c
c     input element data ('input___')
c
      write(*,'(A)') "subroutine flux1"
      call flux1(a(mp(mshl  )),a(mp(mw    )),
     &           a(mp(mc    )),a(mp(mgrav )),
     &           a(mp(mien  )),a(mp(mmat  )),
     &           a(mpid      ),a(mp(mlm   )),
     &           a(mpdiag    ),a(mp(mipar )),
     &           a(mpx       ),a(mp(mlado)),
     &           a(mp(mshlc )),a(mp(mwc   )),
     &           a(mp(mshlpn)),a(mp(mwpn  )),
     &           a(mp(mshln )),a(mp(mwn   )),
     &           a(mp(mshlb )),a(mp(mshlp )),
     &           a(mp(mwp   )),a(mp(mdside)),
     &           a(mp(mshsde)),
c
     &           a(mp(mshlpsd)),a(mp(mshlcsd)),
c
     &           ntype ,numel ,numat ,
     &           nint  ,nrowsh,nesd  ,
     &           nen   ,ndof  ,ned   ,
     &           iprtin,numnp ,ncon  ,
     &           nencon,necon ,nints ,
     &           nnods ,nenlad,npars ,
     &           nenp  ,nside ,nodsp )
c
      write(*,*) "fim da flux1"
      return
c
  200 continue
c
c     ('form_stb')
c
      read(iin,*) index,iwrite
c
c     Esta rotina calcula projecoes locais:
c     para a formulacao hibrida primal
c
      write(*,'(A)') "subroutine flux0primal"
      call flux0primal(a(mp(mien  )),a(mpx       ),a(mp(mxl   )),
     &           a(mpd       ),a(mp(mdl   )),a(mp(mmat  )),
     &           a(mp(mdet  )),a(mp(mshl  )),a(mp(mshg  )),
     &           a(mp(mw    )),a(mp(mc    )),
     &           a(mp(mgrav )),a(mp(mipar )),a(mp(mlado )),
     &           a(mp(mdetc )),a(mp(mshlc )),a(mp(mshgc )),
     &           a(mp(melefd)),a(mp(melred)),a(mp(mshln )),
     &           a(mp(mshgn )),a(mp(mwn   )),a(mp(mdetn )),
     &           a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
     &           a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
     &           a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
     &           a(mp(mdsfl )),a(mp(mddis )),a(mp(mdetp )),
     &           a(mp(mshlp )),a(mp(mshgp )),
c
     &           a(mp(melma )),a(mp(melmb )),a(mp(melmc )),
     &           a(mp(melmd )),a(mp(melmh )),a(mp(melmbb)),
     &           a(mp(melmcb)),a(mp(melmhb)),a(mp(melfa )),
     &           a(mp(melfb )),a(mp(melfc )),a(mp(melfd )),
     &           a(mp(melfab)),a(mp(melfbb)),a(mp(melfcb)),
     &           a(mp(melmdb)),
c
     &           a(mp(mshsde)),
     &           a(mp(mshlpsd)),a(mp(mshlcsd)),
c
     &           a(mp(mshgpsd)),a(mp(mshgcsd)),
c
c
     &           numel ,neesq ,nen   ,
     &           nsd   ,nesd  ,nint  ,
     &           neg   ,nrowsh,ned   ,
     &           nee   ,numnp ,ndof  ,
     &           ncon  ,nencon,necon ,
     &           neep  ,nints ,nnods ,
     &           nenlad, npars,nside ,
     &           nenp  ,nodsp ,index ,nface, nedge)

      write(*,'(A)') "fim da flux0primal"
      write(*,*) " "
      write(*,*) " "
c
c     escreve malha GMSH
c
c$$$      write(*,'(A)') "escreve malha.msh"
c$$$      call dumpmsh(a(mp(mien)),a(mpx),a(mp(mxl)),a(mp(mdside)),
c$$$     &             a(mp(mlado)),numel,numnp,nen,nsd,nesd,
c$$$     &             nedge,npars,nside)
c
c     Esta rotina calcula os erros da projecao local
c     para a formulacao hibrida primal
c
      write(*,'(A)') "subroutine flnormp (depois da flux0)"
      call flnormp(a(mp(mien )),a(mpx       ),a(mp(mxl   )),
     &             a(mpd      ),a(mp(mdl   )),a(mp(mmat  )),
     &             a(mp(mc   )),a(mp(mipar )),a(mp(mdlf )) ,
     &             a(mp(mdlp )),a(mp(mdsfl )),a(mp(mdet  )),
     &             a(mp(mshl )),a(mp(mshg  )),a(mp(mw    )),
     &             a(mp(mdetc )),a(mp(mshlc)),a(mp(mshgc )),
     &             a(mp(mddis )),a(mp(mdetp)),a(mp(mshlp )),
     &             a(mp(mshgp )),
c
     &             a(mp(mshln )),a(mp(mshgn )),
     &             a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
     &             a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
     &             a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
     &             a(mp(mgrav )),a(mp(mwn   )),
c
     &             numel ,neesq ,nen   ,nsd   ,
     &             nesd  ,nint  ,neg   ,nrowsh,
     &             ned   ,nee   ,numnp ,ndof  ,
     &             ncon  ,nencon,necon ,index ,
     &             nints ,ilp ,nenp  ,
     &             nside ,nnods ,nenlad,npars ,
     &             nmultp,nodsp )
      write(*,*) "NODSP",nodsp
      write(*,'(A)') "fim da flnormp"
c
c  esta rotina calcula os erros das interpolantes
c
c$$$      write(*,'(A)') "subroutine flninter"
c$$$      call flninter(a(mp(mien )),a(mpx       ),a(mp(mxl   )),
c$$$     &            a(mpd      ),a(mp(mdl   )),a(mp(mmat  )),
c$$$     &            a(mp(mc   )),a(mp(mipar )),a(mp(mdlf )) ,
c$$$     &            a(mp(mdlp )),a(mp(mdsfl )),a(mp(mdet  )),
c$$$     &            a(mp(mshl )),a(mp(mshg  )),a(mp(mw    )),
c$$$     &            a(mp(mdetc )),a(mp(mshlc)),a(mp(mshgc )),
c$$$     &            a(mp(mddis )),a(mp(mdetp)),a(mp(mshlp )),
c$$$     &            a(mp(mshgp )),
c$$$c
c$$$     &            a(mp(mshln )),a(mp(mshgn )),
c$$$     &            a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
c$$$     &            a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
c$$$     &            a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
c$$$     &            a(mp(mgrav )),a(mp(mwn   )),
c$$$c
c$$$     &            numel ,neesq ,nen   ,nsd   ,
c$$$     &            nesd  ,nint  ,neg   ,nrowsh,
c$$$     &            ned   ,nee   ,numnp ,ndof  ,
c$$$     &            ncon  ,nencon,necon ,index ,
c$$$     &            nints ,interpl,nenp  ,
c$$$     &            nside ,nnods ,nenlad,npars ,
c$$$     &            nmultp,nodsp )
c
c
      return
c
 300  continue
c
c     Esta rotina monta a matriz global (multiplicadores
c     usando condensacao estatica dos graus de liberdade
c     da variavel primalx
c
      write(*,'(A)') "subroutine flux2"
      call flux2(a(mp(mien  )),a(mpx       ),a(mp(mxl   )),
     &           a(mpd       ),a(mp(mdl   )),a(mp(mmat  )),
     &           a(mp(mdet  )),a(mp(mshl  )),a(mp(mshg  )),
     &           a(mp(mw    )),a(mp(mc    )),a(mpalhs    ),
     &           a(mpbrhs    ),a(mpdiag    ),a(mp(mlm   )),
     &           a(mp(mgrav )),a(mp(mipar )),a(mp(mlado )),
     &           a(mp(mdetc )),a(mp(mshlc )),a(mp(mshgc )),
     &           a(mp(melefd)),a(mp(melred)),a(mp(mshln )),
     &           a(mp(mshgn )),a(mp(mwn   )),a(mp(mdetn )),
     &           a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
     &           a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
     &           a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
     &           a(mp(mdsfl )),a(mp(mddis )),a(mp(mdetp )),
     &           a(mp(mshlp )),a(mp(mshgp )),a(mpdlhs   ),
c
     &           a(mp(melma )),a(mp(melmb )),a(mp(melmc )),
     &           a(mp(melmd )),a(mp(melmh )),a(mp(melmbb)),
     &           a(mp(melmcb)),a(mp(melmhb)),a(mp(melfa )),
     &           a(mp(melfb )),a(mp(melfc )),a(mp(melfd )),
     &           a(mp(melfab)),a(mp(melfbb)),a(mp(melfcb)),
     &           a(mp(melmdb)),a(mp(melmbc)),
c
     &           a(mp(mshsde)),a(mped      ),
c
     &           a(mp(mshlpsd)),a(mp(mshlcsd)),
     &           a(mp(mshgpsd)),a(mp(mshgcsd)),
c
     &           numel ,neesq ,nen   ,nsd   ,
     &           nesd  ,nint  ,neg   ,nrowsh,
     &           ned   ,nee   ,numnp ,ndof  ,
     &           ncon  ,nencon,necon ,neep  ,
     &           nints ,nnods ,nenlad,npars ,
     &           nside ,nenp  ,nedge ,nodsp ,
     &           index ,nface)
c
      write(*,'(A)') "fim da flux2"
      return
c
  400 continue
c
c  esta rotina calcula as aproximacoes locai para a variavel
c  primal p no nivel de cada elemento usando as aproximacoes do multiplicador
c
      write(*,'(A)') "subroutine flux3primal"
      call flux3primal(a(mp(mien  )),a(mpx       ),a(mp(mxl   )),
     &           a(mpd       ),a(mp(mdl   )),a(mp(mmat  )),
     &           a(mp(mdet  )),a(mp(mshl  )),a(mp(mshg  )),
     &           a(mp(mw    )),a(mp(mc    )),
     &           a(mp(mgrav )),a(mp(mipar )),a(mp(mlado )),
     &           a(mp(mdetc )),a(mp(mshlc )),a(mp(mshgc )),
     &           a(mp(melefd)),a(mp(melred)),a(mp(mshln )),
     &           a(mp(mshgn )),a(mp(mwn   )),a(mp(mdetn )),
     &           a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
     &           a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
     &           a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
     &           a(mp(mdsfl )),a(mp(mddis )),a(mp(mdetp )),
     &           a(mp(mshlp )),a(mp(mshgp )),
c
     &           a(mp(melma )),a(mp(melmb )),a(mp(melmc )),
     &           a(mp(melmd )),a(mp(melmh )),a(mp(melmbb)),
     &           a(mp(melmcb)),a(mp(melmhb)),a(mp(melfa )),
     &           a(mp(melfb )),a(mp(melfc )),a(mp(melfd )),
     &           a(mp(melfab)),a(mp(melfbb)),a(mp(melfcb)),
     &           a(mp(melmdb)),
c
     &           a(mp(mshsde)),
c
     &           a(mp(mshlpsd)),a(mp(mshlcsd)),
     &           a(mp(mshgpsd)),a(mp(mshgcsd)),
c
     &           numel ,neesq ,nen   ,
     &           nsd   ,nesd  ,nint  ,
     &           neg   ,nrowsh,ned   ,
     &           nee   ,numnp ,ndof  ,
     &           ncon  ,nencon,necon ,
     &           neep  ,nints ,nnods ,
     &           nenlad,npars ,nside ,
     &           nenp  ,nodsp ,index ,nface)

      write(*,'(A)') "fim da flux3primal"
c
c  esta rotina calcula os erros das aproximacoes para a variavel
c  primal (p) no nivel de cada elemento e multiplicador (grad u)
c
      write(*,'(A)') "subroutine flnormp (depois da flux3primal)"
      write(*,*) "NODSP",nodsp
      call flnormp(a(mp(mien )),a(mpx       ),a(mp(mxl   )),
     &            a(mpd      ),a(mp(mdl   )),a(mp(mmat  )),
     &            a(mp(mc   )),a(mp(mipar )),a(mp(mdlf )) ,
     &            a(mp(mdlp )),a(mp(mdsfl )),a(mp(mdet  )),
     &            a(mp(mshl )),a(mp(mshg  )),a(mp(mw    )),
     &            a(mp(mdetc )),a(mp(mshlc)),a(mp(mshgc )),
     &            a(mp(mddis )),a(mp(mdetp)),a(mp(mshlp )),
     &            a(mp(mshgp )),
c
     &            a(mp(mshln )),a(mp(mshgn )),
     &            a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
     &            a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
     &            a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
     &            a(mp(mgrav )),a(mp(mwn   )),
c
     &            numel ,neesq ,nen   ,nsd   ,
     &            nesd  ,nint  ,neg   ,nrowsh,
     &            ned   ,nee   ,numnp ,ndof  ,
     &            ncon  ,nencon,necon ,index ,
     &            nints ,ipp,   nenp  ,
     &            nside ,nnods ,nenlad,npars ,
     &            nmultp,nodsp )
      
      write(*,'(A)') "fim da flnormp"      

c$$$  c
c$$$      call gnusol(a(mp(mien )),a(mpx       ),a(mp(mxl   )),
c$$$     &            a(mpd      ),a(mp(mdl   )),a(mp(mmat  )),
c$$$     &            a(mp(mc   )),a(mp(mipar )),a(mp(mdlf )) ,
c$$$     &            a(mp(mdlp )),a(mp(mdsfl )),a(mp(mdet  )),
c$$$     &            a(mp(mshl )),a(mp(mshg  )),a(mp(mw    )),
c$$$     &            a(mp(mdetc )),a(mp(mshlc)),a(mp(mshgc )),
c$$$     &            a(mp(mddis )),a(mp(mdetp)),a(mp(mshlp )),
c$$$     &            a(mp(mshgp )),
c$$$c
c$$$     &            a(mp(mshln )),a(mp(mshgn )),
c$$$     &            a(mp(mdetb )),a(mp(mshlb )),a(mp(mshgb )),
c$$$     &            a(mp(mdetpn)),a(mp(mshlpn)),a(mp(mshgpn)),
c$$$     &            a(mp(mdside)),a(mp(mxls  )),a(mp(midlsd)),
c$$$     &            a(mp(mgrav )),a(mp(mwn   )),
c$$$c
c$$$     &            numel ,neesq ,nen   ,nsd   ,
c$$$     &            nesd  ,nint  ,neg   ,nrowsh,
c$$$     &            ned   ,nee   ,numnp ,ndof  ,
c$$$     &            ncon  ,nencon,necon ,index ,
c$$$     &            nints ,iwrite,ipp ,nenp  ,
c$$$     &            nside ,nnods ,nenlad,npars ,
c$$$     &            nmultp,nodsp )
c
      return
      end

c-----------------------------------------------------------------------
      subroutine flux1(shl   ,w     ,
     &                 c     ,grav  ,
     &                 ien   ,mat   ,
     &                 id    ,lm    ,
     &                 idiag ,ipar  ,
     &                 x     ,lado  ,
     &                 shlc  ,wc    ,
     &                 shlpn ,wpn   ,
     &                 shln  ,wn    ,
     &                 shlb  ,shlp  ,
     &                 wp    ,idside,
c
     &                 shsde ,
c
     &                 shlpsd,shlcsd,
c
     &                 ntype ,numel ,numat ,
     &                 nint  ,nrowsh,nesd  ,
     &                 nen   ,ndof  ,ned   ,
     &                 iprtin,numnp ,ncon  ,
     &                 nencon,necon ,nints ,
     &                 nnods,nenlad ,npars ,
     &                 nenp ,nside  ,nodsp)
c-----------------------------------------------------------------------
c     program to read, generate and write data for the
c     four-node quadrilateral, elastic continuum element
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension c(10,*),grav(*),ien(nen,*),mat(*),
     &          id(ndof,*),lm(ned,nodsp,*),idiag(*),
     &          ipar(nodsp,*),idside(nside,*),
     &          x(nesd,*),lado(nside,*)
c
      dimension shlb (nrowsh,nenlad,*)
      dimension shln (nrowsh,nnods,*),wn(*)
      dimension shlpn(nrowsh,npars,*),wpn(*)
c
      dimension shl (nrowsh+1,nen,*),w(*)
      dimension shlp(nrowsh+1,nenp,*),wp(*)
      dimension shlc(nrowsh+1,nencon,*),wc(*)
c
      dimension shsde (nrowsh+1,nen,*)
      dimension shlpsd(nrowsh+1,nenp,*)
      dimension shlcsd(nrowsh+1,nencon,*)
c
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
      common /colhtc / neq
c
      write(*,'(A,I5)') " nesd", nesd
c
      write(ieco,1000) ntype,numel,numat
      write(ieco,2000) nint
c
c     handle 2D problems
c
      if(nesd.eq.2) then
c
c     geomeria 1D / Lagrange
c     generation of local shape functions and weight values
c
         nintb=nside*nints
c
         write(*,'(A)') " subroutine flux1 - 1d local shape "
         call oneshl(shlb,wp,nints,nenlad)
         call oneshl(shln,wn,nints,nnods)
         call oneshl(shlpn,wpn,nints,npars)
c
         write(*,'(A)') " subroutine flux1 - 2d local shape tri/quad"
         if(nen.eq.3.or.nen.eq.6.or.nen.eq.10) then
            call shlt(shl,w,nint,nen)
            call shlt(shlc,wc,nint,nencon)
            call shlt(shlp,wp,nint,nenp)
c
            call shltpbk(shlpsd,nenp,nside,nints)
            call shltpbk(shlcsd,nencon,nside,nints)
            call shltpbk(shsde,nen,nside,nints)
         else
            call shlq(shl,w,nint,nen)
            call shlqpk(shlc,wc,nint,nencon)
            call shlqpk(shlp,wp,nint,nenp)
c
            call shlqpbk(shlpsd,nenp,nside,nnods,nints)
            call shlqpbk(shlcsd,nencon,nside,nnods,nints)
            call shlqpbk(shsde,nen,nside,nenlad,nints)
         end if
c
      else if(nesd.eq.3) then
         write(*,'(A)') " funcoes de forma shl 2d e 3d"
         write(*,'(A,10I5)') " nint, nen    ", nint, nen
         write(*,'(A,10I5)') " nint, nencon ", nint, nencon
         write(*,'(A,10I5)') " nint, nenp   ", nint, nenp
         write(*,'(A,10I5)') " nenp, nside  ",nenp,nside
         write(*,'(A,10I5)') " nencon,nside ",nencon,nside
         write(*,'(A,10I5)') " nnods,nints  ",nnods, nints
         write(*,'(A,10I5)') " nrowsh,nenlad",nrowsh,nenlad
         write(*,'(A,10I5)') " nenlad,nnods,npars",nenlad,nnods,npars
c
c     treat faces first (2D)
c     obs: shlq and shlqpk compute the same functions
c
         write(*,'(A)') " calculando shl do multiplicador"
         write(*,'(A,3I5)') " nenlad,nnods,npars",nenlad,nnods,npars

         write(*,*)
         write(*,'(A)') " calculando shlb"
         call shlqpk(shlb,wp,nints,nenlad)

         write(*,*)
         write(*,'(A)') " calculando shln"
         call shlqpk(shln,wn,nints,nnods)

         write(*,*)
         write(*,'(A)') " calculando shlpn"
         call shlqpk(shlpn,wpn,nints,npars)

c     *** DEBUG ***
c     shp do multiplicador (dim 2)
         write(*,*)
         write(*,'(A,2I5)') " dados shln", nnods, nints
         do ii=1,nints
            write(*,999) (shln(3,in,ii),in=1,nnods)
         end do
         write(*,*)

         write(*,'(A,2I5)') " dados shlpn", npars, nints
         do ii=1,nints
            write(*,999) (shlpn(3,in,ii),in=1,npars)
         end do
         write(*,*)
c
c     then 3d stuff
c     calcular: shl, shlc, shlp, shlpsd, shlcsd, shlsde
c
         write(*,'(A)') "calculando shl da variavel"
         write(*,'(A,2I5)') "nint,nen",nint,nen

         write(*,*)
         write(*,'(A)') " calculando shl"
         call shlhxpk(shl,w,nint,nen)

         write(*,*)
         write(*,'(A)') " calculando shlc"
         call shlhxpk(shlc,wc,nint,nencon)

         write(*,*)
         write(*,'(A)') " calculando shlp"
         call shlhxpk(shlp,wp,nint,nenp)
c
c     shp on faces
c
         nenlad2=2
         nnods2=2

         if(nints.eq.4) then
            nints2=2
         else if(nints.eq.9) then
            nints2=3
         end if
c
         write(*,*)
         write(*,'(A)')  " integral em face/area"
         write(*,'(A,2I5)') " nints2,nen2",nints2,nnods2
c
         write(*,*)
         write(*,'(A)') " calculando shlpsd (shlhxpbk)"
         call shlhxpbk(shlpsd,nenp,nside,nnods2,nints2)

         write(*,*)
         write(*,'(A)') " calculando shlcsd (shlhxpbk)"
         call shlhxpbk(shlcsd,nencon,nside,nnods2,nints2)

         write(*,*)
         write(*,'(A)') " calculando shsde (shlhxpbk)"
         call shlhxpbk(shsde,nen,nside,nenlad2,nints2)

c     *** DEBUG ***
c     shp on volumes
         write(*,*) "pesos da integracao de gauss - w"
         write(*,'(10F8.4)') (w(ii),ii=1,nint)
         write(*,*) "pesos da integracao de gauss - wc"
         write(*,'(10F8.4)') (wc(ii),ii=1,nint)
         write(*,*) "pesos da integracao de gauss - wp"
         write(*,'(10F8.4)') (wp(ii),ii=1,nint)

         write(*,*) "shl", nint, nen
         do ii=1,nint
            write(*,999) (shl(4,in,ii),in=1,nen)
         end do

         write(*,*) "shlc", nint, nencon
         do ii=1,nint
            write(*,999) (shlc(4,in,ii),in=1,nencon)
         end do

         write(*,*) "shlp", nint, nenp
         do ii=1,nint
            write(*,999) (shlp(4,in,ii),in=1,nenp)
         end do

c     *** DEBUG ***
c     shp on faces
         write(*,*)
         write(*,*) "dados shlpsd - on faces"

         write(*,*) "pesos da integracao de gauss - wn"
         write(*,'(10F8.4)') (wn(ii),ii=1,nints)
         write(*,*) "pesos da integracao de gauss - wpn"
         write(*,'(10F8.4)') (wpn(ii),ii=1,nints)

         write(*,*) "shlpsd"
         do ii=1,nints*nside
            write(*,999) (shlpsd(4,in,ii),in=1,nenp)
         end do
         write(*,*) "shlcsd",nencon,nside
         do ii=1,nints*nside
            write(*,999) (shlcsd(4,in,ii),in=1,nencon)
         end do
         write(*,*) "shsde"
         do ii=1,nints*nside
            write(*,999) (shsde(4,in,ii),in=1,nen)
         end do
c
 999     format(' ',30F10.6)
c     *** DEBUG ***
      end if
c
      nintb=nside*nints
c
c     read material properties
c
      call fluxmx(c,numat)
c
c     constant body forces
c
      read (iin,5000) (grav(i),i=1,3)
      write (ieco,6000) (grav(i),i=1,3)
c
c     generation of conectivities
c
      write(*,'(A)') "subroutine genel"
      call genel(ien,mat,nen)
c
      if (iprtin.eq.0) call prntel(mat,ien,nen,numel)
c
c     generation of conectivety for element multipliers
c
      write(*,'(A)') "subroutine genside"
      call genside(idside,nside,nencon)
      write(*,*) "array idside"
      do il=1,6
         write(*,*) il, (idside(il,j),j=1,4)
      end do

      write(*,'(A)') "subroutine genelad"
      call genelad(lado,nside)
      if (iprtin.eq.0) call prntels(mat,lado,nside,numel)
c
      write(*,*) "DEBUG LADO(,)"
      do kk=1,numel
         write(*,*) (lado(ll,kk),ll=1,6)
      end do
c
      write(*,'(A)') "subroutine genelpar"
      call genelpar(ipar,ien,lado,idside,
     &     nen,nside,nodsp,numel,npars)

c
      if (iprtin.eq.0) call prntelp(mat,ipar,nodsp,numel)
c
c     generation of lm array
c
      call formlm(id,ipar,lm,ndof,ned,nodsp,numel)
c
c     modification of idiag array
c
      call colht(idiag,lm,ned,nodsp,numel,neq)
c
c     Neumann BCs not implemented YET
c
      return
c
 1000 format(///,
     &' d u a l   h i b r i d   m i x e d   f o r m u l a t i o n ',///
     &//5x,' element type number . . . . . . . . . . .(ntype ) = ',i10
     &//5x,' number of elements  . . . . . . . . . . .(numel ) = ',i10
     &//5x,' number of element material sets . . . . .(numat ) = ',i10)
2000  format(
     &//5x,' numerical integration points  . . . . . .(nint  ) = ',i10)
 5000 format(8f10.0)
 6000 format(////' ',
     &' g r a v i t y   v e c t o r   c o m p o n e n t s      ',//5x,
     &' W-1 direction (sinxsiny)  . . . . . . . = ',      1pe15.8//5x,
     &' W-2 direction (cosxcosy)  . . . . . . . = ',      1pe15.8//5x,
     &' W-3 direction (Polinomial). . . . . . . = ',      1pe15.8//5x)
c
      end

c-----------------------------------------------------------------------
      subroutine genside(idside,nside,nen)
c-----------------------------------------------------------------------
c     program to read and generate element node and material numbers
c        idside(nside,nnods) = element sides
c-----------------------------------------------------------------------
      dimension idside(nside,*)
      common /info/ iexec,iprtin,irank,nsd,numnp,ndof,nlvect,
     &     numeg,nmultp,nedge
c
c     2D definitions
c
      if(nsd.eq.2) then
c
c     define idside
c
           if(nen.eq.4) then
              idside(1,1) = 1
              idside(1,2) = 2
c
              idside(2,1) = 2
              idside(2,2) = 3
c
              idside(3,1) = 3
              idside(3,2) = 4
c
              idside(4,1) = 4
              idside(4,2) = 1
           end if
c
            if(nen.eq.9) then
              idside(1,1) = 1
              idside(1,2) = 2
              idside(1,3) = 5
c
              idside(2,1) = 2
              idside(2,2) = 3
              idside(2,3) = 6
c
              idside(3,1) = 3
              idside(3,2) = 4
              idside(3,3) = 7
c
              idside(4,1) = 4
              idside(4,2) = 1
              idside(4,3) = 8
             end if
c
           if(nen.eq.16) then
c  lado 1
              idside(1,1) = 1
              idside(1,2) = 2
              idside(1,3) = 5
              idside(1,4) = 6
c   lado 2
              idside(2,1) = 2
              idside(2,2) = 3
              idside(2,3) = 7
              idside(2,4) = 8
c
c   lado 3
c
              idside(3,1) = 3
              idside(3,2) = 4
              idside(3,3) = 9
              idside(3,4) = 10
c
c   lado 4
c
              idside(4,1) = 4
              idside(4,2) = 1
              idside(4,3) = 11
              idside(4,4) = 12
c
c
           end if
c
           if(nen.eq.25) then
c
c     lado 1
c
              idside(1,1) = 1
              idside(1,2) = 2
              idside(1,3) = 5
              idside(1,4) = 6
              idside(1,5) = 7
c
c     lado 2
c
              idside(2,1) = 2
              idside(2,2) = 3
              idside(2,3) = 8
              idside(2,4) = 9
              idside(2,5) = 10
c
c     lado 3
c
              idside(3,1) = 3
              idside(3,2) = 4
              idside(3,3) = 11
              idside(3,4) = 12
              idside(3,5) = 13

c
c     lado 4
c
              idside(4,1) = 4
              idside(4,2) = 1
              idside(4,3) = 14
              idside(4,4) = 15
              idside(4,5) = 16

c
           end if
c
c
           if(nen.eq.36) then
c
c     lado 1
c
              idside(1,1) = 1
              idside(1,2) = 2
              idside(1,3) = 5
              idside(1,4) = 6
              idside(1,5) = 7
              idside(1,6) = 8
c
c     lado 2
c
              idside(2,1) = 2
              idside(2,2) = 3
              idside(2,3) = 9
              idside(2,4) = 10
              idside(2,5) = 11
              idside(2,6) = 12
c
c     lado 3
c
              idside(3,1) = 3
              idside(3,2) = 4
              idside(3,3) = 13
              idside(3,4) = 14
              idside(3,5) = 15
              idside(3,6) = 16

c
c     lado 4
c
              idside(4,1) = 4
              idside(4,2) = 1
              idside(4,3) = 17
              idside(4,4) = 18
              idside(4,5) = 19
              idside(4,6) = 20
c
           end if
c
c
           if(nen.eq.49) then
c
c     lado 1
c
              idside(1,1) = 1
              idside(1,2) = 2
              idside(1,3) = 5
              idside(1,4) = 6
              idside(1,5) = 7
              idside(1,6) = 8
              idside(1,7) = 9
c
c     lado 2
c
              idside(2,1) = 2
              idside(2,2) = 3
              idside(2,3) = 10
              idside(2,4) = 11
              idside(2,5) = 12
              idside(2,6) = 13
              idside(2,7) = 14
c
c     lado 3
c
              idside(3,1) = 3
              idside(3,2) = 4
              idside(3,3) = 15
              idside(3,4) = 16
              idside(3,5) = 17
              idside(3,6) = 18
              idside(3,7) = 19

c
c     lado 4
c
              idside(4,1) = 4
              idside(4,2) = 1
              idside(4,3) = 20
              idside(4,4) = 21
              idside(4,5) = 22
              idside(4,6) = 23
              idside(4,7) = 24
c
           end if
c
c
           if(nen.eq.64) then
c
c     lado 1
c
              idside(1,1) = 1
              idside(1,2) = 2
              idside(1,3) = 5
              idside(1,4) = 6
              idside(1,5) = 7
              idside(1,6) = 8
              idside(1,7) = 9
              idside(1,8) = 10
c
c     lado 2
c
              idside(2,1) = 2
              idside(2,2) = 3
              idside(2,3) = 11
              idside(2,4) = 12
              idside(2,5) = 13
              idside(2,6) = 14
              idside(2,7) = 15
              idside(2,8) = 16
c
c     lado 3
c
              idside(3,1) = 3
              idside(3,2) = 4
              idside(3,3) = 17
              idside(3,4) = 18
              idside(3,5) = 19
              idside(3,6) = 20
              idside(3,7) = 21
              idside(3,8) = 22

c
c     lado 4
c
              idside(4,1) = 4
              idside(4,2) = 1
              idside(4,3) = 23
              idside(4,4) = 24
              idside(4,5) = 25
              idside(4,6) = 26
              idside(4,7) = 27
              idside(4,8) = 28
c
           end if
c
c
          if(nen.eq.3) then
             idside(1,1) = 1
             idside(1,2) = 2
             idside(2,1) = 2
             idside(2,2) = 3
             idside(3,1) = 3
             idside(3,2) = 1
          end if
c
          if(nen.eq.6) then
             idside(1,1) = 1
             idside(1,2) = 2
             idside(1,3) = 4

             idside(2,1) = 2
             idside(2,2) = 3
             idside(2,3) = 5

             idside(3,1) = 3
             idside(3,2) = 1
             idside(3,3) = 6
          end if
c
          if(nen.eq.10) then
c
             idside(1,1) = 1
             idside(1,2) = 2
             idside(1,3) = 4
             idside(1,4) = 5
c
             idside(2,1) = 2
             idside(2,2) = 3
             idside(2,3) = 6
             idside(2,4) = 7
c
             idside(3,1) = 3
             idside(3,2) = 1
             idside(3,3) = 8
             idside(3,4) = 9
c
          end if
       end if
c
c     3d definitions
c
       if (nsd.eq.3) then
c
          if(nen.eq.8) then
c     face 1
            idside(1,1) = 1
            idside(1,2) = 2
            idside(1,3) = 6
            idside(1,4) = 5
c     face 2
            idside(2,1) = 1
            idside(2,2) = 4
            idside(2,3) = 8
            idside(2,4) = 5
c     face 3
            idside(3,1) = 2
            idside(3,2) = 3
            idside(3,3) = 7
            idside(3,4) = 6
c     face 4
            idside(4,1) = 4
            idside(4,2) = 3
            idside(4,3) = 7
            idside(4,4) = 8
c     face 5
            idside(5,1) = 1
            idside(5,2) = 2
            idside(5,3) = 3
            idside(5,4) = 4
c     face 6
            idside(6,1) = 5
            idside(6,2) = 6
            idside(6,3) = 7
            idside(6,4) = 8
         end if
c
       end if
c
       return
       end

c-----------------------------------------------------------------------
      subroutine flux0primal(ien   ,x     ,xl    ,
     &                       d     ,dl    ,mat   ,
     &                       det   ,shl   ,shg   ,
     &                       w     ,c     ,
     &                       grav  ,ipar  ,lado  ,
     &                       detc  ,shlc  ,shgc  ,
     &                       eleffd,elresd,shln  ,
     &                       shgn  ,wn    ,detn  ,
     &                       detb  ,shlb  ,shgb  ,
     &                       detpn ,shlpn ,shgpn ,
     &                       idside,xls   ,idlsd ,
     &                       dsfl  ,ddis  ,detp  ,
     &                       shlp  ,shgp  ,
     &                       elma  ,elmb  ,elmc  ,
     &                       elmd  ,elmh  ,elmbb ,
     &                       elmcb ,elmhb ,elfa  ,
     &                       elfb  ,elfc  ,elfd  ,
     &                       elfab ,elfbb ,elfcb ,
     &                       elmdb ,
c
     &                       shsde ,
c
     &                       shlpsd,shlcsd,
     &                       shgpsd,shgcsd,
c
     &                       numel ,neesq ,nen   ,
     &                       nsd   ,nesd  ,nint  ,
     &                       neg   ,nrowsh,ned   ,
     &                       nee   ,numnp ,ndof  ,
     &                       ncon  ,nencon,necon ,
     &                       neep  ,nints ,nnods ,
     &                       nenlad,npars ,nside ,
     &                       nenp  ,nodsp ,index ,nface, nedge)
c-----------------------------------------------------------------------
c     program to calculate stifness matrix and force array for the
c        plane elasticity element and
c        assemble into the global left-hand-side matrix
c        and right-hand side vector
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical diag,quad,hexa
c
      dimension elma(necon,*),elmb(necon,*),elmc(necon,*),elmd(neep,*),
     &          elmh(neep,*),elmbb(neep,*),elmcb(neep,*),elmhb(nee,*)
      dimension elmdb(nee,*)
      dimension elfa(*),elfb(*),elfc(*),elfd(*),elfab(*),
     &          elfbb(*),elfcb(*)
      dimension ien(nen,*),x(nsd,*),xl(nesd,*),d(ndof,*),dl(ned,*),
     &          mat(*),det(*),w(*),c(10,*),
     &          grav(*),ipar(nodsp,*),lado(nside,*),detc(*),
     &          eleffd(nee,*),elresd(*)
      dimension wn(*),detn(*),detb(*),detpn(*),
     &          idside(nside,*),xls(nesd,*),idlsd(*),
     &          dsfl(ncon,nencon,*),ddis(ned,nenp,*)
      dimension dls(12),detp(*)
c
      dimension shlb (nrowsh,nenlad,*),shgb (nrowsh,nenlad,*)
      dimension shln (nrowsh,nnods,*), shgn (nrowsh,nnods,*)
      dimension shlpn(nrowsh,npars,*), shgpn(nrowsh,npars,*)
c
      dimension shl (nrowsh+1,nen,*),   shg (nrowsh+1,nen,*)
      dimension shlp(nrowsh+1,nenp,*),  shgp(nrowsh+1,nenp,*)
      dimension shlc(nrowsh+1,nencon,*),shgc(nrowsh+1,nencon,*)
c
      dimension shsde (nrowsh+1,nen,*)
      dimension shlpsd(nrowsh+1,nenp,*),shlcsd(nrowsh+1,nencon,*)
      dimension shgpsd(nrowsh+1,nenp,*),shgcsd(nrowsh+1,nencon,*)
c
      dimension solex(nen)
      dimension vab(3),vad(3),vc(3),xn(3),coo(3), xxn(3)
c
      integer elfaces(nedge,nenlad)
      integer iboolf(nedge)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
c     consistent matrix
c
      write(*,*) "NROWSH=",nrowsh
      write(*,*) "NROWSH+1=",nrowsh+1
c
      diag = .false.
      pi=4.d00*datan(1.d00)
      gf1=grav(1)
      gf2=grav(2)
      gf3=grav(3)

c     ******
c     TODO: CHAMAR A FUNCAO PRO GMSH
c     call dumpmsh(...)
c     ******

c
c     dump mesh in screen / GMSH file
c
      do i=1,nedge
         iboolf(i)=0
      end do
c
      open(123, file='malha.msh')
      write(*,*) "dump mesh to screen/file"
      write(123,'(a)') "$MeshFormat"
      write(123,'(a)') "2.2 0 8"
      write(123,'(a)') "$EndMeshFormat"
      write(123,'(a)') "$Nodes$"

      write(*,*) " coordenadas", nen, nsd, nesd
      do np=1,numnp
         write(*,456) np, (x(i,np), i=1,nsd)
      end do
      write(123,*) numnp
c     2D
      if(nsd.eq.2) then
         do np=1,numnp
            write(123,456) np, (x(i,np), i=1,nsd), 0.0
         end do
      end if
c     3D
      if(nsd.eq.3) then
         do np=1,numnp
            write(123,456) np, (x(i,np), i=1,nsd)
         end do
      end if
c
 456  format(' ',I6,30F10.6)
      write(123,'(a)') "$EndNodes"

      write(123,'(a)') "$Elements"
      write(123,*) nedge+numel
      write(*,*) " conectividade - elementos"
      do nel=1,numel
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
         write(*,*) nel, (ien(j, nel), j=1,nen)
      end do
      write(*,*) " faces por elemento"
      do nel=1,numel
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
         write(*,*) nel, (lado(j,nel), j=1,nside)
      end do
      write(*,*) " numeracao local das faces"
      do il=1,nside
         write(*,*) il, (idside(il,j),j=1,4)
      end do
      write(*,*)
      write(*,*) " conectividade - faces"
c
c     2D
c
      if(nsd.eq.2) then
         kel = 1
         do nel=1,numel
            call local(ien(1,nel),x,xl,nen,nsd,nesd)
            do ns=1,nside
               ns1=idside(ns,1)
               ns2=idside(ns,2)
c
               nl1=ien(ns1,nel)
               nl2=ien(ns2,nel)
c
               iface=lado(ns,nel)
               write(*,*) nel, ns, iface, nl1, nl2
               if(iboolf(iface).eq.0) then
                  iboolf(iface) = 1
                  elfaces(iface,1) = nl1
                  elfaces(iface,2) = nl2
               end if
               kel = kel + 1
            end do
         end do
         do ifc=1,nedge
            nl1=elfaces(ifc,1)
            nl2=elfaces(ifc,2)
            write(123,789) ifc, 1, 2, 100, 200, nl1, nl2
         end do
         kel = nedge + 1
         do nel=1,numel
            call local(ien(1,nel),x,xl,nen,nsd,nesd)
            write(123,789) nedge+nel,3,2,100,200,(ien(j,nel),j=1,nen)
         end do
      end if
c
c     3D
c
      if(nsd.eq.3) then
         kel = 1
         do nel=1,numel
            call local(ien(1,nel),x,xl,nen,nsd,nesd)
            do ns=1,nside
               ns1=idside(ns,1)
               ns2=idside(ns,2)
               ns3=idside(ns,3)
               ns4=idside(ns,4)
c
               nl1=ien(ns1,nel)
               nl2=ien(ns2,nel)
               nl3=ien(ns3,nel)
               nl4=ien(ns4,nel)
c
               iface=lado(ns,nel)
               write(*,*) nel, ns, iface, nl1, nl2, nl3, nl4
               if(iboolf(iface).eq.0) then
                  iboolf(iface) = 1
                  elfaces(iface,1) = nl1
                  elfaces(iface,2) = nl2
                  elfaces(iface,3) = nl3
                  elfaces(iface,4) = nl4
               end if
               kel = kel + 1
            end do
         end do
         do ifc=1,nedge
            nl1=elfaces(ifc,1)
            nl2=elfaces(ifc,2)
            nl3=elfaces(ifc,3)
            nl4=elfaces(ifc,4)
            write(123,789) ifc, 3, 2, 100, 200, nl1, nl2, nl3, nl4
         end do
         kel = nedge + 1
         do nel=1,numel
            call local(ien(1,nel),x,xl,nen,nsd,nesd)
            write(123,789) nedge+nel,5,2,100,200,(ien(j,nel),j=1,nen)
         end do
      end if
c
      write(123,'(a)') "$EndElements"
      close(123)
 789  format(' ', 16I5)
c     fim do output no GMSH

c     **************************************************************************
c     element loop
c     **************************************************************************
      do 500 nel=1,numel

         write(*,*) ""
         write(*,*) ""
         write(*,*) "ELEMENTO",nel
c
c     clear stiffness matrix and force array
c
         call clear(elmbb,neep*neep)
         call clear(elfbb,neep)
c
c     localize coordinates and Dirichlet b.c.
c
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
c
c     centroide
c
         call centroid(xl,nsd,nen,coo)
c
c     identify problem 2D or 3D and element type
c
         if(nsd.eq.2) then
            m = mat(nel)
            quad = .true.
            if (nen.eq.4.and.ien(3,nel).eq.ien(4,nel)) quad = .false.
c
            call shgqs(xl,detc,shlc,shgc,nint,nel,neg,quad,nencon,
     &           shl,nen)
            call shgqs(xl,detp,shlp,shgp,nint,nel,neg,quad,nenp,shl,nen)
c
            nintb=nside*nints
c
            call shgtqsd(ien(1,nel),xl,shlpsd,shgpsd,
     &           nside,nintb,nints,nel,neg,nenp,shsde,nen)
            call shgtqsd(ien(1,nel),xl,shlcsd,shgcsd,
     &           nside,nintb,nints,nel,neg,nencon,shsde,nen)
c
c     call shgqsd(xl,shlpsd,shgpsd,nintb,nel,neg,nenp,shl,nen)
c     call shgqsd(xl,shlcsd,shgcsd,nintb,nel,neg,nencon,shl,nen)
c
         else if(nsd.eq.3) then
c
            m = mat(nel)
            quad = .false.
            hexa = .true.
            write(*,*) " subroutines shg"
            write(*,*) " subroutines shghx1 (shgc)"
            call shghx(xl,detc,shlc,shgc,nint,nel,neg,hexa,nencon,
     &           shl,nen)
            write(*,*) " subroutines shghx2 (shgp)"
            call shghx(xl,detp,shlp,shgp,nint,nel,neg,hexa,nenp,
     &           shl,nen)

c     NINTB eh o total de pontos de integracao por elemento
c     ex: em 2D temos 4 lados x 2 pt de int em cada aresta = 8
c     ex: em 3D sao 6 faces x 4 pts por face = 24 pontos

            nintb=nside*nints
            write(*,'(A,10I5)') " nintb,nints,nside=",nintb,nints,nside
c
            write(*,'(A)') " subroutines shghxd 1 (shgpsd)"
            call shgthxsd(ien(1,nel),xl,shlpsd,shgpsd,
     &           nside,nintb,nints,nel,neg,nenp,shsde,nen)

            write(*,'(A)') " subroutines shghxd 2 (shgcsd)"
            call shgthxsd(ien(1,nel),xl,shlcsd,shgcsd,
     &           nside,nintb,nints,nel,neg,nencon,shsde,nen)
         end if
c
c     form stiffness matrix
c

c
c     calculo de h - caracteristico para cada elemento
c
         if(nen.eq.3.or.nen.eq.6) then
            h2=xl(1,2)*xl(2,3)+xl(1,1)*xl(2,2)+xl(1,3)*xl(2,1)
     &           -xl(1,1)*xl(2,3)-xl(1,2)*xl(2,1)-xl(1,3)*xl(2,2)
            h2=h2*pt5
            h=dsqrt(h2)
         else
            h2=0
            do l=1,2
               h2=h2+(xl(1,l)-xl(1,l+2))**2+(xl(2,l)-xl(2,l+2))**2
            end do
            h=dsqrt(h2)/2.d00
            h2=h*h
         end if
c
c     set up material properties
c
         betah = c(7,m)*h**c(8,m)
         eps = c(1,m)
         b1 = c(2,m)
         b2 = c(3,m)
         b3 = 0.d0
         upwind = c(4,m)
         write(*,'(A,F8.4)') " h ",h
         write(*,'(A,F8.4)') " beta   ",c(7,m)
         write(*,'(A,F8.4)') " beta/h ",betah
c
c     loop on integration points
c
         do l=1,nint
            c1=detc(l)*w(l)
c
c     f = gf1*sin(pix)*sin(piy) + gf2*(Prob2)
c
            xx = 0.d00
            yy = 0.d00
            zz = 0.d00
c
            do i=1,nen
               xx = xx + shl(4,i,l)*xl(1,i)
               yy = yy + shl(4,i,l)*xl(2,i)
               zz = zz + shl(4,i,l)*xl(3,i)
            end do
c
            pix = pi*xx
            piy = pi*yy
            piz = pi*zz
            sx = dsin(pix)
            sy = dsin(piy)
            sz = dsin(piz)
            cx = dcos(pix)
            cy = dcos(piy)
            cz = dcos(piz)
c
            pi2 = pi*pi
            co = 1.d00
c
            pss = gf1*(3.d00*eps*pi2*sx*sy*sz + b1*pi*cx*sy+b2*pi*sx*cy)
     &           + gf2*(-pi*(-dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)*eps
     &           + dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)*eps*dexp((yy -
     &           0.1D1)/eps) + dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)
     &  *eps*dexp((xx - 0.1D1)/eps) - dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)
     &      *eps*dexp((xx - 0.2D1 + yy)/eps) - dcos(pix/0.2D1
     &        )*dsin(piy/0.2D1)*dexp((xx - 0.1D1)/eps) + dcos(pix/0.2D1
     &    )*dsin(piy/0.2D1)*dexp((xx - 0.2D1 + yy)/eps) - dsin(pix/0.2D1
     &      )*dcos(piy/0.2D1)*dexp((yy - 0.1D1)/eps) + dsin(pix/0.2D1
     &           )*dcos(piy/0.2D1)*dexp((xx - 0.2D1 + yy)/eps)
     &           - dcos(pix/0.2D1)*dsin(piy/0.2D1) + dcos(pix/0.2D1)
     &          *dsin(piy/0.2D1)*dexp((yy - 0.1D1)/eps) - dsin(pix/0.2D1
     &           )*dcos(piy/0.2D1) + dsin(pix / 0.2D1)*dcos(piy/0.2D1
     &           )*dexp((xx - 0.1D1)/eps))/0.2D1)
     &           + gf3*1.d00
c
c     loop to compute volume integrals
c
            do j=1,nenp
               djx=shgp(1,j,l)*c1
               djy=shgp(2,j,l)*c1
               djz=shgp(3,j,l)*c1
               djn=shgp(4,j,l)*c1
c
c     source terms
c
               nbj=ned*(j-1)
               nbj1=nbj+1
c
               elfbb(nbj1) = elfbb(nbj1) + djn*pss
c
c     element stiffness
c
               do i=1,nenp
                  nbi=ned*(i-1)
                  nbi1=nbi+1
c
                  dix=shgp(1,i,l)
                  diy=shgp(2,i,l)
                  diz=shgp(3,i,l)
                  din=shgp(4,i,l)
c
c     (grad p, grad q) + (b.grad p,q) = (f,q)
c
                  elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                             + eps*(dix*djx + diy*djy + diz*djz)
     &                             + (b1*djx + b2*djy + b3*djz)*din
               end do
            end do
         end do
c
c     debug - stiffness and load vector
c
         write(*,*) "matriz (poisson)"
         do ik=1,neep
            write(*,'(20F10.6)') (elmbb(ik,jk),jk=1,neep)
         end do
         write(*,*) "vetor (poisson)"
         write(*,'(20F10.6)') (elfbb(jk),jk=1,neep)

c     **************************************************************************
c     *** loop to compute area integrals - symmetrization - boundary terms   ***
c     **************************************************************************

         do 4000 ns=1,nside
            write(*,*) " "
            write(*,'(A,I5)') "face ", ns
c
c     localiza os no's do lado ns
c
            ns1=idside(ns,1)
            ns2=idside(ns,2)
            ns3=idside(ns,3)
            ns4=idside(ns,4)

            nl1=ien(ns1,nel)
            nl2=ien(ns2,nel)
            nl3=ien(ns3,nel)
            nl4=ien(ns4,nel)

            write(*,'(A,4I5)') " nodes", nl1, nl2, nl3, nl4
c
c$$$            if(nl2.gt.nl1) then
c$$$               sign = 1.d00
c$$$               do  nn=1,nenlad
c$$$                  idlsd(nn)=idside(ns,nn)
c$$$               end do
c$$$            else
c$$$               sign = -1.d00
c$$$               idlsd(1) = idside(ns,2)
c$$$               idlsd(2) = idside(ns,1)
c$$$               id3 = nenlad-2
c$$$               if(id3.gt.0) then
c$$$                  do il=id3,nenlad
c$$$                     idlsd(il) = idside(ns,nenlad+id3-il)
c$$$                  end do
c$$$               end if
c$$$            end if
c
c     calcula sign para a aresta/face
c
            call vec3sub(xl(1,ns2), xl(1,ns1), vab)
            call vec3sub(xl(1,ns4), xl(1,ns1), vad)
            call vec3sub(coo,xl(1,ns1),vc)
            call cross(vab,vad,xn)
            call nrm3(xn)
            dotcn = dot3(vc,xn)

            if(dotcn.lt.0.d0) then
               sign = 1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
            else
               write(*,'(A)') " trocou sinal"
               sign = -1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
c
c               idlsd(1) = idside(ns,4)
c               idlsd(2) = idside(ns,3)
c               idlsd(3) = idside(ns,2)
c               idlsd(4) = idside(ns,1)

c               id3 = nenlad-2
c               if(id3.gt.0) then
c                  do il=id3,nenlad
c                     idlsd(il) = idside(ns,nenlad+id3-il)
c                  end do
c               end if
            end if
            write(*,'(A,F8.4)') " sinal", sign
c
c     confere nos globais depois que trocou
c
            ns1=idlsd(1)
            ns2=idlsd(2)
            ns3=idlsd(3)
            ns4=idlsd(4)

            nl1=ien(ns1,nel)
            nl2=ien(ns2,nel)
            nl3=ien(ns3,nel)
            nl4=ien(ns4,nel)
            write(*,'(A,4I5)') " nodes globais", nl1,nl2,nl3,nl4
c
c     ajeita a normal com o sinal
c
            xn(1)=sign*xn(1)
            xn(2)=sign*xn(2)
            xn(3)=sign*xn(3)
c
c     localize the coordinates of the side nodes
c
            do nn=1,nenlad
               nl=idlsd(nn)
               xls(1,nn)=xl(1,nl)
               xls(2,nn)=xl(2,nl)
               xls(3,nn)=xl(3,nl)
            end do
c
            ndgs = lado(ns,nel)
c
            if(nesd.eq.2) then
c
c     problema 2D
c
               call oneshgp(xls,detn,shlb,shln,shgn,
     &              nenlad,nnods,nints,nesd,ns,nel,neg)
               call oneshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad,npars,nints,nesd,ns,nel,neg)
            else
c
c     problema 3D
c
               nenlad2=4
               npars2=npars
               nnods2=4
ccc               call twoshgp(xls,detn,shlb,shln,shgn,
ccc     &              nenlad2,nnods2,nints,nesd,ns,nel,neg,xxn)
               call twoshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad2,npars2,nints,nesd,ns,nel,neg,xxn)
            end if
c
c     projecao local do multiplicador
c
            call drchbc3(shgpn,shlb,detpn,wn,gf1,gf2,gf3,xls,dls,
     &           eps,pi,nints,nenlad,npars)
c
c     confere projecao local do multiplicador
c
            write(*,'(A)') " proj l2 local "
            do i=1,nenlad
               axx = xls(1,i)
               ayy = xls(2,i)
               azz = xls(3,i)
               sx=dsin(pi*axx)
               sy=dsin(pi*ayy)
               sz=dsin(pi*azz)
               pi2=pi*pi
               co=1.d00/(pi2*2.d00+gamas)
               dhex = gf1*sx*sy*sz
               solex(i) = dhex
            end do
            write(*,'(A,4I8)')    " idlsd ", (idlsd(i),i=1,npars)
            write(*,'(A,10F8.4)') " proje ", (dls(i),i=1,npars)
            write(*,'(A,10F8.4)') " exata ", (solex(i),i=1,nenlad)
c
c     TODO: AQUI TEM QUE VER O QUE ESTA ROLANDO
c     tinha um problema de guardar 2x vezes no mesmo lugar
c     por causa de elementos que compartilham a mesma face
c     fiz a gambiarra abaixo de guardar com ordem trocada de acordo com sinal
c
            ngs = npars*(ndgs-1)
            nls = npars*(ns-1)

            if(sign.gt.0) then
               do i=1,npars
                  write(*,'(A,F8.4,2I5)') " save ",dls(i),i,ngs+i
                  d(1,ngs + i) = dls(i)
               end do
c$$$            else
c$$$               do i=npars,1,-1
c$$$                  indx=ngs+(npars+1-i)
c$$$                  write(*,'(A,F8.4,2I5)') " save ",dls(i),i,indx
c$$$                  d(1,indx) = dls(i)
c$$$               end do
            end if

            write(*,'(A,3F8.4)') " normal xn ",xn(1),xn(2),xn(3)
c            write(*,'(A,3F8.4)') " normal xxn",xxn(1),xxn(2),xxn(3)

c
c     compute boundary integral - symmetrization
c
            do 1000 ls = 1,nints
c
               lb = (ns-1)*nints + ls
c
c     geometria
c
               x1 = 0.d00
               x2 = 0.d00
               x3 = 0.d00
               do i=1,nenlad
                  x1 = x1 + xls(1,i)*shlb(3,i,ls)
                  x2 = x2 + xls(2,i)*shlb(3,i,ls)
                  x3 = x3 + xls(3,i)*shlb(3,i,ls)
               end do
c               write(*,'(A,3F8.4)') "x1,x2,x3",x1,x2,x3

c
c     normal da face
c
               xn1 = xn(1)
               xn2 = xn(2)
               xn3 = xn(3)
c               write(*,'(A,3F8.4)') " normal xn in",xn(1),xn(2),xn(3)

c
c     valores exatos dos multiplicadores
c
               pix=pi*x1
               piy=pi*x2
               piz=pi*x3
               sx=dsin(pix)
               sy=dsin(piy)
               sz=dsin(piz)
               cx=dcos(pix)
               cy=dcos(piy)
               cz=dcos(piz)
c
               pi2=pi*pi
               xinveps = eps**(-1.0)
               co=1.d00
c
c     valor exato do multiplicador
c
               dhs = gf1*sx*sy*sz
     &          + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*(1.d00 - dexp((x1
     &          - 1.d00)/eps))*(1.d00 - dexp((x2 - 1.d00)/eps)))
     &          + gf3*1.d00
c
               bn = b1*xn1 + b2*xn2 + b3*xn3
c
c               write(*,'(A,8F8.4)') "psd 1: ", (shgpsd(1,j,lb),j=1,nenp)
c               write(*,'(A,8F8.4)') "psd 2: ", (shgpsd(2,j,lb),j=1,nenp)
c               write(*,'(A,8F8.4)') "psd 3: ", (shgpsd(3,j,lb),j=1,nenp)
c               write(*,'(A,8F8.4)') "psd 4: ", (shgpsd(4,j,lb),j=1,nenp)
c               write(*,*)
c
               do j=1,nenp
                  nbj = ned*(j-1)
                  nbj1=nbj+1
                  djx=shgpsd(1,j,lb)*detpn(ls)*wn(ls)
                  djy=shgpsd(2,j,lb)*detpn(ls)*wn(ls)
                  djz=shgpsd(3,j,lb)*detpn(ls)*wn(ls)
                  djn=shgpsd(4,j,lb)*detpn(ls)*wn(ls)
c
                  gjn = djx*xn1 + djy*xn2 + djz*xn3
c
                  elfbb(nbj1) = elfbb(nbj1)
     &                        + eps*(betah*dhs*djn - dhs*gjn)
     &                        + upwind*dhs*max(0.d00,-bn)*djn
c
                  do i=1,nenp
                     nbi =ned*(i-1)
                     nbi1=nbi+1
                     dix=shgpsd(1,i,lb)
                     diy=shgpsd(2,i,lb)
                     diz=shgpsd(3,i,lb)
                     din=shgpsd(4,i,lb)
c
                     gin = dix*xn1 + diy*xn2 + diz*xn3
c
                     elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                                - eps*(gin*djn + din*gjn)
     &                                + eps*betah*din*djn
     &                                + upwind*din*max(0.d00,-bn)*djn
                  end do
               end do
c
 1000       continue
 4000    continue

c
c     projecao local primal
c
         write(*,*)
         write(*,'(A)') " call to solvetdg - proj local primal"
         write(*,'(A,I5)') " size", neep
         write(*,'(A)') " matriz"
         do ik=1,neep
            write(*,'(20F8.4)') (elmbb(ik,jk),jk=1,neep)
         end do
         write(*,'(A)') " vetor"
         write(*,'(20F8.4)') (elfbb(jk),jk=1,neep)

         call solvetdg(elmbb,elfbb,neep)

         write(*,'(A)') " solucao"
         write(*,'(20F8.4)') (elfbb(jk),jk=1,neep)
c
c     projecao local do potencial
c
         do j=1,nenp
            ddis(1,j,nel) = elfbb(j)
         end do
c
c     confere valor nodal da flux0
c
         write(*,*)
         write(*,'(A)') " confere valor da flux0 - x,y,z,ex,fl0,err"
         do i=1,nenp
            xx = xl(1,i)
            yy = xl(2,i)
            zz = xl(3,i)
            sx = sin(pi*xx)
            sy = sin(pi*yy)
            sz = sin(pi*zz)
            exx = sx*sy*sz
            write(*,'(10F8.4)') xx, yy, zz, exx, elfbb(i),
     &                                  dabs(exx-elfbb(i))
         end do
c
 500  continue

c
c      nm = nedge*npars
c      write(*,*)
c      write(*,*) "array de multiplicadores - pos flux0"
c      do kk=1,nm
c         write(*,'(I5,F8.4)') kk, d(1,kk)
c         if(mod(kk,4).eq.0) then
c            write(*,*) " "
c         end if
c      end do
c
      return
      end

c------------------------------------------------------------------------------
      subroutine flux2(ien   ,x     ,xl    ,
     &                 d     ,dl    ,mat   ,
     &                 det   ,shl   ,shg   ,
     &                 w     ,c     ,alhs  ,
     &                 brhs  ,idiag ,lm    ,
     &                 grav  ,ipar  ,lado  ,
     &                 detc  ,shlc  ,shgc  ,
     &                 eleffd,elresd,shln  ,
     &                 shgn  ,wn    ,detn  ,
     &                 detb  ,shlb  ,shgb  ,
     &                 detpn ,shlpn ,shgpn ,
     &                 idside,xls   ,idlsd ,
     &                 dsfl  ,ddis  ,detp  ,
     &                 shlp  ,shgp  ,dlhs,
     &                 elma  ,elmb  ,elmc  ,
     &                 elmd  ,elmh  ,elmbb ,
     &                 elmcb ,elmhb ,elfa  ,
     &                 elfb  ,elfc  ,elfd  ,
     &                 elfab ,elfbb ,elfcb ,
     &                 elmdb ,elmbc ,
c
     &                 shsde ,ideg  ,
c
     &                 shlpsd,shlcsd,
     &                 shgpsd,shgcsd,
c
     &                 numel ,neesq ,nen   ,nsd   ,
     &                 nesd  ,nint  ,neg   ,nrowsh,
     &                 ned   ,nee   ,numnp ,ndof  ,
     &                 ncon  ,nencon,necon ,neep  ,
     &                 nints ,nnods ,nenlad,npars ,
     &                 nside ,nenp  ,nedge ,nodsp ,index,nface)
c------------------------------------------------------------------------------
c     program to calculate stifness matrix and force array
c     for ...
c------------------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical diag,quad,hexa,zerodl
c
      dimension elma(necon,*),elmb(necon,*),elmc(necon,*),elmd(neep,*)
      dimension elmh(neep,*),elmbb(neep,*),elmcb(neep,*),elmhb(nee,*)
      dimension elmdb(nee,*),elmbc(nee,*)
      dimension elfa(*),elfb(*),elfc(*),elfd(*),elfab(*)
      dimension elfbb(*),elfcb(*)
c
      dimension ien(nen,*),x(nsd,*),xl(nesd,*),mat(*)
      dimension d(ndof,*),dl(ned,*),dls(12)
      dimension w(*),c(10,*),alhs(*),brhs(*),idiag(*),lm(ned,nodsp,*),
     &     grav(*),ipar(nodsp,*),lado(nside,*),dlhs(*),
     &     detc(*),eleffd(nee,*),elresd(*)
      dimension wn(*),idside(nside,*),xls(nesd,*),idlsd(*),
     &     dsfl(ncon,nencon,*),ddis(ned,nenp,*)
      dimension det(*),detp(*),detb(*),detpn(*)
      dimension ideg(2*ndof,*)
c
      dimension shlb (nrowsh,nenlad,*),shgb (nrowsh,nenlad,*)
      dimension shln (nrowsh,nnods,*), shgn (nrowsh,nnods,*)
      dimension shlpn(nrowsh,npars,*), shgpn(nrowsh,npars,*)
c
      dimension shl (nrowsh+1,nen,*),   shg (nrowsh+1,nen,*)
      dimension shlp(nrowsh+1,nenp,*),  shgp(nrowsh+1,nenp,*)
      dimension shlc(nrowsh+1,nencon,*),shgc(nrowsh+1,nencon,*)
c
      dimension shsde (nrowsh+1,nen,*)
      dimension shlpsd(nrowsh+1,nenp,*),shlcsd(nrowsh+1,nencon,*)
      dimension shgpsd(nrowsh+1,nenp,*),shgcsd(nrowsh+1,nencon,*)
c
      dimension solex(nen),vab(3),vad(3),vc(3),xn(3),coo(3),xxn(3)
c
ccccc    dimension shsde(nside,nen,*),ideg(2*ndof,*)
cccc     dimension shlpsd(nrowsh,nenp,*),shlcsd(nrowsh,nencon,*),
cccc     &     shgpsd(nrowsh,nenp,*),shgcsd(nrowsh,nencon,*)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      write(*,*) "FLUX2"
c
c     consistent matrix
c
      diag = .false.
      pi = 4.d00*datan(1.d00)
      gf1 = grav(1)
      gf2 = grav(2)
      gf3 = grav(3)

c     **************************************************************************
c     element loop
c     **************************************************************************
      do 500 nel=1,numel
c
c     clear stiffness matrix and force array
c
         call clear(eleffd,nee*nee)
         call clear(elmbb,neep*neep)
         call clear(elmcb,nee*neep)
         call clear(elmbc,nee*neep)
c
         call clear(elresd,nee)
         call clear(elfbb,neep)
         call clear(elfb,neep)
         call clear(elfc,nee)
c
c     localize coordinates and Dirichlet BCs
c
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
         call local(ipar(1,nel),d,dl,nodsp,ndof,ned)
c
c     centroide
c
         call centroid(xl,nsd,nen,coo)
c
c     material prop
c         
         m = mat(nel)
c
         if(nsd.eq.2) then
            write(*,*) "PROBLEMA 2D"
            quad = .true.
            if (nen.eq.4.and.ien(3,nel).eq.ien(4,nel)) quad = .false.
c
            call shgqs(xl,detc,shlc,shgc,nint,nel,neg,quad,
     &            nencon,shl,nen)
            call shgqs(xl,detp,shlp,shgp,nint,nel,neg,quad,
     &            nenp,shl,nen)
c
            nintb=nside*nints
c
            call shgtqsd(ien(1,nel),xl,shlpsd,shgpsd,
     &           nside,nintb,nints,nel,neg,nenp,shsde,nen)
            call shgtqsd(ien(1,nel),xl,shlcsd,shgcsd,
     &           nside,nintb,nints,nel,neg,nencon,shsde,nen)
c
c          call shgqsd(xl,shlpsd,shgpsd,nintb,nel,neg,nenp,shl,nen)
c          call shgqsd(xl,shlcsd,shgcsd,nintb,nel,neg,nencon,shl,nen)
c
         else if(nsd.eq.3) then
            write(*,*) "PROBLEMA 3D"
            quad = .false.
            hexa = .true.
            write(*,*) " subroutines shg"
            write(*,*) " subroutines shghx1 (shgc)"
            call shghx(xl,detc,shlc,shgc,nint,nel,neg,hexa,nencon,
     &           shl,nen)
            write(*,*) " subroutines shghx2 (shgp)"
            call shghx(xl,detp,shlp,shgp,nint,nel,neg,hexa,nenp,
     &           shl,nen)
c
            nintb=nside*nints
c
            write(*,'(A,10I5)') " nintb,nints,nside=",nintb,nints,nside
c
            write(*,'(A)') " subroutines shghxd 1 (shgpsd)"
            call shgthxsd(ien(1,nel),xl,shlpsd,shgpsd,
     &           nside,nintb,nints,nel,neg,nenp,shsde,nen)

            write(*,'(A)') " subroutines shghxd 2 (shgcsd)"
            call shgthxsd(ien(1,nel),xl,shlcsd,shgcsd,
     &           nside,nintb,nints,nel,neg,nencon,shsde,nen)
c
         end if
c
c     form stiffness matrix
c

c
c     calculo de h - caracteristico para cada elemento
c
         if(nen.eq.3.or.nen.eq.6) then
            h2=xl(1,2)*xl(2,3)+xl(1,1)*xl(2,2)+xl(1,3)*xl(2,1)
     &           -xl(1,1)*xl(2,3)-xl(1,2)*xl(2,1)-xl(1,3)*xl(2,2)
            h2=h2*pt5
            h=dsqrt(h2)
         else
            h2=0
            do l=1,2
               h2=h2+(xl(1,l)-xl(1,l+2))**2+(xl(2,l)-xl(2,l+2))**2
            end do
            h=dsqrt(h2)/2.d00
            h2=h*h
         end if
c
c     set up material properties
c
         betah = c(7,m)*h**c(8,m)
         eps = c(1,m)
         b1 = c(2,m)
         b2 = c(3,m)
         b3 = 0.d0
         upwind = c(4,m)
c
c     loop on integration points
c
         write(*,*) "FLUX2 int points"
         do l=1,nint
            c1=detc(l)*w(l)
c
c     f = gf1*sin(pix)*sin(piy) + gf2*cos(pix)*cos(piy)
c
            xx = 0.d00
            yy = 0.d00
            zz = 0.d00
c
            do i=1,nen
               xx = xx + shl(4,i,l)*xl(1,i)
               yy = yy + shl(4,i,l)*xl(2,i)
               zz = zz + shl(4,i,l)*xl(3,i)
            end do
c
            pix = pi*xx
            piy = pi*yy
            piz = pi*zz
            sx = dsin(pix)
            sy = dsin(piy)
            sz = dsin(piz)
            cx = dcos(pix)
            cy = dcos(piy)
            cz = dcos(piz)
c
            pi2 = pi*pi
            co = 1.d00
c
            pss = 3.d00*eps*pi2*sx*sy*sz
c
c$$$            pss = gf1*(3.d00*eps*pi2*sx*sy*sz + b1*pi*cx*sy+b2*pi*sx*cy)
c$$$     &           + gf2*(-pi*(-dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)*eps
c$$$     &           + dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)*eps*dexp((yy -
c$$$     &           0.1D1)/eps) + dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)
c$$$     &  *eps*dexp((xx - 0.1D1)/eps) - dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)
c$$$     &           *eps*dexp((xx - 0.2D1 + yy)/eps) - dcos(pix/0.2D1
c$$$     &         )*dsin(piy/0.2D1)*dexp((xx - 0.1D1)/eps) + dcos(pix/0.2D1
c$$$     &    )*dsin(piy/0.2D1)*dexp((xx - 0.2D1 + yy)/eps) - dsin(pix/0.2D1
c$$$     &        )*dcos(piy/0.2D1)*dexp((yy - 0.1D1)/eps) + dsin(pix/0.2D1
c$$$     &           )*dcos(piy/0.2D1)*dexp((xx - 0.2D1 + yy)/eps)
c$$$     &           - dcos(pix/0.2D1)*dsin(piy/0.2D1) + dcos(pix/0.2D1)
c$$$     &         *dsin(piy/0.2D1)*dexp((yy - 0.1D1)/eps) - dsin(pix/0.2D1
c$$$     &           )*dcos(piy/0.2D1) + dsin(pix / 0.2D1)*dcos(piy/0.2D1
c$$$     &           )*dexp((xx - 0.1D1)/eps))/0.2D1)
c$$$  &           + gf3*1.d00
            
c     **************************************************************************
c     loop to compute volume integrals
c     **************************************************************************
            do j=1,nenp
               djx=shgp(1,j,l)*c1
               djy=shgp(2,j,l)*c1
               djz=shgp(3,j,l)*c1
               djn=shgp(4,j,l)*c1
c
c     source terms
c
               nbj=ned*(j-1)
               nbj1=nbj+1
c
c     (grad p , grad q) + (b.grad p,q) = (f,q)
c
               elfbb(nbj1) = elfbb(nbj1) + djn*pss
c
c     element stiffness
c
               do i=1,nenp
                  nbi=ned*(i-1)
                  nbi1=nbi+1
c
                  dix=shgp(1,i,l)
                  diy=shgp(2,i,l)
                  diz=shgp(3,i,l)
                  din=shgp(4,i,l)
c
c     (grad p, grad q) + (b.grad p,q) = (f,q)
c
                  elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                             + eps*(dix*djx + diy*djy + diz*djz)
     &                             + (b1*djx + b2*djy + b3*djz)*din
               end do
            end do
         end do
         
c     **************************************************************************
c     boundary terms
c     **************************************************************************
         
         do 4000 ns=1,nside
c
c     localiza os no's do lado ns
c
            ns1=idside(ns,1)
            ns2=idside(ns,2)
            ns3=idside(ns,3)
            ns4=idside(ns,4)

            nl1=ien(ns1,nel)
            nl2=ien(ns2,nel)
            nl3=ien(ns3,nel)
            nl4=ien(ns4,nel)
c
c$$$            if(nl2.gt.nl1) then
c$$$               sign = 1.d00
c$$$               do  nn=1,nenlad
c$$$                  idlsd(nn)=idside(ns,nn)
c$$$               end do
c$$$            else
c$$$               sign = -1.d00
c$$$               idlsd(1) = idside(ns,2)
c$$$               idlsd(2) = idside(ns,1)
c$$$               id3 = nenlad-2
c$$$               if(id3.gt.0) then
c$$$                  do il=id3,nenlad
c$$$                     idlsd(il) = idside(ns,nenlad+id3-il)
c$$$                  end do
c$$$               end if
c$$$            end if
c
c
c     calcula sign para a aresta/face
c
            call vec3sub(xl(1,ns2), xl(1,ns1), vab)
            call vec3sub(xl(1,ns4), xl(1,ns1), vad)
            call vec3sub(coo,xl(1,ns1),vc)
            call cross(vab,vad,xn)
            call nrm3(xn)
            dotcn = dot3(vc,xn)

            if(dotcn.lt.0.d0) then
               sign = 1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
            else
               write(*,'(A)') " trocou sinal"
               sign = -1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
c
c               idlsd(1) = idside(ns,4)
c               idlsd(2) = idside(ns,3)
c               idlsd(3) = idside(ns,2)
c               idlsd(4) = idside(ns,1)

c               id3 = nenlad-2
c               if(id3.gt.0) then
c                  do il=id3,nenlad
c                     idlsd(il) = idside(ns,nenlad+id3-il)
c                  end do
c               end if
            end if
            write(*,'(A,F8.4)') " sinal", sign
c
c     confere nos globais depois que trocou (ou nao)
c
            ns1=idlsd(1)
            ns2=idlsd(2)
            ns3=idlsd(3)
            ns4=idlsd(4)

            nl1=ien(ns1,nel)
            nl2=ien(ns2,nel)
            nl3=ien(ns3,nel)
            nl4=ien(ns4,nel)
            write(*,'(A,4I5)') " nodes globais", nl1,nl2,nl3,nl4
c
c     ajeita a normal com o sinal
c
            xn(1)=sign*xn(1)
            xn(2)=sign*xn(2)
            xn(3)=sign*xn(3)
c
c     localize the coordinates of the side nodes
c
            do nn=1,nenlad
               nl=idlsd(nn)
               xls(1,nn)=xl(1,nl)
               xls(2,nn)=xl(2,nl)
               xls(3,nn)=xl(3,nl)
            end do
c
c     problema 2d
c
            if(nesd.eq.2) then
               call oneshgp(xls,detn,shlb,shln,shgn,
     &              nenlad,nnods,nints,nesd,ns,nel,neg)
               call oneshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad,npars,nints,nesd,ns,nel,neg)
c
            else if(nesd.eq.3) then
c
c     problema 3d
c
               nenlad2=4
               npars2=npars
               nnods2=4
ccc               call twoshgp(xls,detn,shlb,shln,shgn,
ccc     &              nenlad2,nnods2,nints,nesd,ns,nel,neg,xxn)
               call twoshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad2,npars2,nints,nesd,ns,nel,neg,xxn)
            end if
c
c     Dirichlet boundary conditions on the multiplier
c
            ndgs = lado(ns,nel)
c
            if(ideg(1,ndgs).eq.1) then
               write(*,*) "ELEM",nel,"FACE",ns,"NDGS",ndgs,"BC OK"
               call drchbc3(shgpn,shlb,detpn,wn,gf1,gf2,gf3,xls,dls,
     &              eps,pi,nints,nenlad,npars)
c               call drchbc(shgpn,shlb,detpn,wn,gf1,gf2,gf3,xls,dls,
c     &                      eps,pi,nints,nenlad,npars)               
               ngs = npars*(ndgs-1)
               nls = npars*(ns-1)
               do i=1,npars
                  d(1,ngs + i) = dls(i)
                  dl(1,nls + i) = dls(i)
               end do
            end if
c
            if(ideg(2,ndgs).eq.1) then
               write(*,*) "IDEG(2,ndgs)=1 ....DUALBC"
               call dualbc(shgpn,shlb,detpn,wn,gf1,gf2,gf3,xls,
     &               elresd,eps,pi,sign,nints,nenlad,ns,npars)
            end if
            
c
c     compute boundary integral
c
            do 1000 ls=1,nints
c
               lb = (ns-1)*nints + ls
c
c     geometria
c
               x1 = 0.d00
               x2 = 0.d00
               x3 = 0.d00
c               dx1 = 0.d00
c               dx2 = 0.d00
c               dx3 = 0.d00
               do i=1,nenlad
                  x1 = x1 + xls(1,i)*shlb(3,i,ls)
                  x2 = x2 + xls(2,i)*shlb(3,i,ls)
                  x3 = x3 + xls(3,i)*shlb(3,i,ls)
c                  dx1 = dx1+xls(1,i)*shlb(1,i,ls)
c                  dx2 = dx2+xls(2,i)*shlb(1,i,ls)
               end do
c               dxx=dsqrt(dx1*dx1+dx2*dx2)
c               xn1= sign*dx2/dxx
c               xn2=-sign*dx1/dxx

c
c     normal da face
c
               xn1 = xn(1)
               xn2 = xn(2)
               xn3 = xn(3)
c
c     bn = b1*xn1 + b2*xn2
c
               do j=1,nenp
c
                  nbj = ned*(j-1)
                  nbj1 = nbj+1
                  djx = shgpsd(1,j,lb)*detpn(ls)*wn(ls)
                  djy = shgpsd(2,j,lb)*detpn(ls)*wn(ls)
                  djz = shgpsd(3,j,lb)*detpn(ls)*wn(ls)
                  djn = shgpsd(4,j,lb)*detpn(ls)*wn(ls)
                  gjn = djx*xn1 + djy*xn2 + djz*xn3
c
                  do i=1,nenp
                     nbi = ned*(i-1)
                     nbi1 = nbi+1
                     dix = shgpsd(1,i,lb)
                     diy = shgpsd(2,i,lb)
                     diz = shgpsd(3,i,lb)
                     din = shgpsd(4,i,lb)
                     gin = dix*xn1 + diy*xn2 + diz*xn3
c
                     elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                                - eps*(gin*djn + din*gjn)
                  end do
               end do
c
c     TODO: descrever...monta matriz XXX ?
c
               do j=1,npars
c
                  ncj1 = (ns-1)*npars + j
                  djn = shgpn(3,j,ls)*detpn(ls)*wn(ls)
c
c     no source term
c
                  do i=1,nenp
                     nbi = ned*(i-1)
                     nbi1 = nbi+1
c
                     dix = shgpsd(1,i,lb)
                     diy = shgpsd(2,i,lb)
                     diz = shgpsd(3,i,lb)
                     din = shgpsd(4,i,lb)
                     gin = dix*xn1 + diy*xn2 + diz*xn3
c
                     elmcb(nbi1,ncj1) = elmcb(nbi1,ncj1) + eps*gin*djn
                     elmbc(ncj1,nbi1) = elmbc(ncj1,nbi1) + eps*gin*djn
                  end do
               end do
c
c     penalty terms
c
               bn = b1*xn1 + b2*xn2 + x3*xn3
c
               do j=1,nenp
                  nbj = ned*(j-1)
                  nbj1 = nbj+1
                  djn = shgpsd(4,j,lb)*detpn(ls)*wn(ls)
c
                  do i=1,nenp
                     nbi = ned*(i-1)
                     nbi1 = nbi+1
                     din = shgpsd(4,i,lb)
                     elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                                + eps*betah*din*djn
     &                                + upwind*din*max(0.d00,-bn)*djn
                  end do
               end do
c
c     TODO2: descrever...monta matriz XXX ?
c
               do j=1,npars
                  ncj1 = (ns-1)*npars + j
                  djn = shgpn(3,j,ls)*detpn(ls)*wn(ls)
c
c     no source term
c
                  do i=1,nenp
                     nbi = ned*(i-1)
                     nbi1 = nbi+1
                     din = shgpsd(4,i,lb)
c
                     elmcb(nbi1,ncj1) = elmcb(nbi1,ncj1)
     &                                - eps*betah*din*djn
     &                                - upwind*din*max(0.d00,-bn)*djn
c
                     elmbc(ncj1,nbi1) = elmbc(ncj1,nbi1)
     &                                - eps*betah*din*djn
     &                                - upwind*din*max(0.d00,bn)*djn
                  end do
c
                  do i=1,npars
                     nci1 = (ns-1)*npars + i
                     din = shgpn(3,i,ls)
c
                     eleffd(nci1,ncj1) = eleffd(nci1,ncj1)
     &                                 + eps*betah*din*djn
     &                                 + upwind*din*max(0.d00,bn)*djn
                  end do
               end do
c
 1000       continue
 4000    continue

         
c
c     matriz C
c
c     c      if(nel.eq.1) then
c     c        write(ielmat,*) 'matriz C',nel,nee
c     c        do i=1,nee
c     c          do j=1,neep
c     c              write(ielmat,91) i,j,elmbc(i,j),elmbc(j,i)
c     c            end do
c     c          end do
c     c   91   format(2i5,2e15.6)
c
c     c      end if
c
c
c     matriz B
c
c     c      if(nel.eq.1) then
c     c        write(ielmat,*) 'matriz B',nel,nee
c     c        do i=1,neep
c     c          do j=1,nee
c     c              write(ielmat,91) i,j,elmcb(i,j),elmcb(j,i)
c     c            end do
c     c          end do
c     c   91   format(2i5,2e15.6)
c
c     c      end if
c
c     matriz A(p_h,q_h)
c
c     c      if(nel.eq.1) then
c     c        write(ielmat,*) 'matriz A',nel,nee
c     c        do i=1,neep
c     c          do j=1,neep
c     c              write(ielmat,91) i,j,elmbb(i,j),elmbb(j,i)
c     c            end do
c     c          end do
c     91   format(2i5,2e15.6)
c
c     c      end if

c
c     condensacao estatica
c
         call condtdg(elmbb,elmcb,elmbc,elmdb,elfbb,eleffd,elresd,
     &        neep,nee)

         write(*,*) "FLUX2 ELEMENTO",nel

         if(nel.eq.1) then
            write(ielmat,*) 'matriz do elemento',nel,nee,eps
            do i=1,nee
               do j=1,nee
                  write(ielmat,91) i,j,eleffd(i,j),eleffd(j,i)
               end do
            end do
 91         format(2i5,2e15.6)
c
         end if
c
c     computation of Dirichlet BCs contribution
c
         call ztest(dl,nee,zerodl)
c
         if(.not.zerodl)
     &        call kdbc(eleffd,elresd,dl,nee)
c
c
c     assemble element stifness matrix and force array into global
c     left-hand-side matrix and right-hand side vector
c
         call addlhs(alhs,eleffd,idiag,lm(1,1,nel),nee,diag)
c        call addnsl(alhs,dlhs,eleffd,idiag,lm(1,1,nel),nee,diag)
c
         call addrhs(brhs,elresd,lm(1,1,nel),nee)
c
c
c
 500  continue
c
c
      return
      end

c-----------------------------------------------------------------------
       subroutine condtdg(elmbb,elmcb,elmbc,elmdb,
     &            elfbb,eleffd,elresd,
     &            neep,nee)
c
      implicit real*8 (a-h,o-z)
c
      dimension elmbb(neep,*),elmcb(neep,*),elmdb(nee,*),
     &          eleffd(nee,*),elmbc(nee,*)
      dimension elfbb(*),elresd(*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
c
c   condensa\E7\E3o do sitema
c
c
c   Bb Xb   + Cb Xc    = Fbb
c
c   Bc Xb  + Cc Xc   = Fc
c
c   com elimina\E7\E3o das inciggnitas  Xb
c
      call invmb(elmbb,neep,neep)
c
c    \br(D) = Bc B^{1}
c
      do i=1,nee
      do j=1,neep
        elmdb(i,j) = 0.d00
        do k=1,neep
          elmdb(i,j) = elmdb(i,j) + elmbc(i,k)*elmbb(k,j)
        end do
      end do
      end do
c
c  eleffd = eleffd - Bc B^{1}C
c
      do i=1,nee
      do j=1,nee
        do k=1,neep
          eleffd(i,j) = eleffd(i,j) - elmdb(i,k)*elmcb(k,j)
        end do
      end do
      end do
c
c  elresd = elresd - Bc B^{1}Fb
c
      do i=1,nee
        do k=1,neep
          elresd(i) = elresd(i) - elmdb(i,k)*elfbb(k)
        end do
      end do

      return
c
      end
      
c-------------------------------------------------------------------------------
      subroutine flux3primal(ien   ,x     ,xl    ,
     &                       d     ,dl    ,mat   ,
     &                       det   ,shl   ,shg   ,
     &                       w     ,c     ,
     &                       grav  ,ipar  ,lado  ,
     &                       detc  ,shlc  ,shgc  ,
     &                       eleffd,elresd,shln  ,
     &                       shgn  ,wn    ,detn  ,
     &                       detb  ,shlb  ,shgb  ,
     &                       detpn ,shlpn ,shgpn ,
     &                       idside,xls   ,idlsd ,
     &                       dsfl  ,ddis  ,detp  ,
     &                       shlp  ,shgp  ,
     &                       elma  ,elmb  ,elmc  ,
     &                       elmd  ,elmh  ,elmbb ,
     &                       elmcb ,elmhb ,elfa  ,
     &                       elfb  ,elfc  ,elfd  ,
     &                       elfab ,elfbb ,elfcb ,
     &                       elmdb ,
c
     &                       shsde ,
c
     &                       shlpsd,shlcsd,
     &                       shgpsd,shgcsd,
c
     &                       numel ,neesq ,nen   ,
     &                       nsd   ,nesd  ,nint  ,
     &                       neg   ,nrowsh,ned   ,
     &                       nee   ,numnp ,ndof  ,
     &                       ncon  ,nencon,necon ,
     &                       neep  ,nints ,nnods ,
     &                       nenlad,npars ,nside ,
     &                       nenp  ,nodsp ,index ,nface)
c-------------------------------------------------------------------------------
c     program to calculate stifness matrix and force array for the
c        plane elasticity element and
c        assemble into the global left-hand-side matrix
c        and right-hand side vector
c-------------------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      logical diag,quad,hexa,zerodl
c      
      dimension elma(necon,*),elmb(necon,*),elmc(necon,*),elmd(neep,*)
      dimension elmh(neep,*),elmbb(neep,*),elmcb(neep,*),elmhb(nee,*)
      dimension elmdb(nee,*)
      dimension elfa(*),elfb(*),elfc(*),elfd(*),elfab(*)
      dimension elfbb(*),elfcb(*)
c      
      dimension eleffd(nee,*),elresd(*)      
      dimension ien(nen,*),x(nsd,*),xl(nesd,*),d(ndof,*),dl(ned,*)
      dimension mat(*),c(10,*),grav(*),ipar(nodsp,*),lado(nside,*)
      dimension w(*),wn(*)
      dimension det(*),detc(*),detb(*),detpn(*),detp(*)
      dimension idside(nside,*),xls(nesd,*),idlsd(*)
      dimension dsfl(ncon,nencon,*),ddis(ned,nenp,*),dls(12)
c
c     shln(3,nnods,*),shgn(3,nnods,*),      
c     ,shlb(3,nenlad,*),shgb(3,nenlad,*),
c     ,shlpn(3,npars,*),shgpn(3,npars,*),            

c     shl(nrowsh,nen,*),shg(nrowsh,nen,*),      
c     shlc(nrowsh,nencon,*), shgc(nrowsh,nencon,*)      
c     ,shlp(3,nenp,*),shgp(3,nenp,*)
      
c      dimension shsde(nside,nen,*)
c      dimension shlpsd(nrowsh,nenp,*),shlcsd(nrowsh,nencon,*),
c     &     shgpsd(nrowsh,nenp,*),shgcsd(nrowsh,nencon,*)
c
c
      dimension shlb (nrowsh,nenlad,*),shgb (nrowsh,nenlad,*)
      dimension shln (nrowsh,nnods,*), shgn (nrowsh,nnods,*)
      dimension shlpn(nrowsh,npars,*), shgpn(nrowsh,npars,*)
c
      dimension shl (nrowsh+1,nen,*),   shg (nrowsh+1,nen,*)
      dimension shlp(nrowsh+1,nenp,*),  shgp(nrowsh+1,nenp,*)
      dimension shlc(nrowsh+1,nencon,*),shgc(nrowsh+1,nencon,*)
c
      dimension shsde (nrowsh+1,nen,*)
      dimension shlpsd(nrowsh+1,nenp,*),shlcsd(nrowsh+1,nencon,*)
      dimension shgpsd(nrowsh+1,nenp,*),shgcsd(nrowsh+1,nencon,*) 
c
      dimension vab(3),vad(3),vc(3),xn(3),coo(3),xxn(3)
c      
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c     
c     consistent matrix
c     
      diag = .false.
      pi=4.d00*datan(1.d00)
      gf1=grav(1)
      gf2=grav(2)
      gf3=grav(3)
c     
      do 500 nel=1,numel
c     
c     clear stiffness matrix and force array
c     
         call clear(elmbb,neep*neep)
c     
         call clear(elfbb,neep)
c     
c     localize coordinates and Dirichlet b.c.
c     
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
         call local(ipar(1,nel),d,dl,nodsp,ndof,ned)
c
c     centroide
c
         call centroid(xl,nsd,nen,coo)         
c
c     material prop
c         
         m = mat(nel)
c
c     check dimension 2D / 3D
c         
         if(nsd.eq.2) then
            quad = .true.
            if (nen.eq.4.and.ien(3,nel).eq.ien(4,nel)) quad = .false.
            call shgqs(xl,detc,shlc,shgc,nint,nel,neg,quad,nencon,
     &           shl,nen)
            call shgqs(xl,detp,shlp,shgp,nint,nel,neg,quad,nenp,
     &            shl,nen)
c           call shgq(xl,det,shl,shg,nint,nel,neg,quad,nen)
c            
            nintb=nside*nints
c     
            call shgtqsd(ien(1,nel),xl,shlpsd,shgpsd,
     &           nside,nintb,nints,nel,neg,nenp,shsde,nen)
            call shgtqsd(ien(1,nel),xl,shlcsd,shgcsd,
     &           nside,nintb,nints,nel,neg,nencon,shsde,nen)            
c           call shgqsd(xl,shlpsd,shgpsd,nintb,nel,neg,nenp,shl,nen)
c           call shgqsd(xl,shlcsd,shgcsd,nintb,nel,neg,nencon,shl,nen)
c
         else if(nsd.eq.3) then
            write(*,*) "PROBLEMA 3D"
            quad = .false.
            hexa = .true.
            write(*,*) " subroutines shg"
            write(*,*) " subroutines shghx1 (shgc)"
            call shghx(xl,detc,shlc,shgc,nint,nel,neg,hexa,nencon,
     &           shl,nen)
            write(*,*) " subroutines shghx2 (shgp)"
            call shghx(xl,detp,shlp,shgp,nint,nel,neg,hexa,nenp,
     &           shl,nen)
c
            nintb=nside*nints
            write(*,'(A,10I5)') " nintb,nints,nside=",nintb,nints,nside
c
            write(*,'(A)') " subroutines shghxd 1 (shgpsd)"
            call shgthxsd(ien(1,nel),xl,shlpsd,shgpsd,
     &           nside,nintb,nints,nel,neg,nenp,shsde,nen)

            write(*,'(A)') " subroutines shghxd 2 (shgcsd)"
            call shgthxsd(ien(1,nel),xl,shlcsd,shgcsd,
     &           nside,nintb,nints,nel,neg,nencon,shsde,nen)
c            
         end if        
c     
c     form stiffness matrix
c

c     
c     calculo de h - caracteristico para cada elemento
c     
         if(nen.eq.3.or.nen.eq.6) then
            h2 =  xl(1,2)*xl(2,3)+xl(1,1)*xl(2,2)+xl(1,3)*xl(2,1)
     &           -xl(1,1)*xl(2,3)-xl(1,2)*xl(2,1)-xl(1,3)*xl(2,2)
            h2 = h2*pt5
            h = dsqrt(h2)
         else
            h2 = 0
            do l=1,2
               h2 = h2+(xl(1,l)-xl(1,l+2))**2+(xl(2,l)-xl(2,l+2))**2
            end do
            h = dsqrt(h2)/2.d00
            h2 = h*h
c     
         end if
c     
c     set up material properties
c     
         betah = c(7,m)*h**c(8,m)
         eps = c(1,m)
         b1 = c(2,m)
         b2 = c(3,m)
         b3 = 0.d0
         upwind = c(4,m)
c     
c     loop on integration points
c
         write(*,*) "FLUX3 int points"
         do l=1,nint
            c1=detc(l)*w(l)
c     
c     f = gf1*sin(pix)*sin(piy) + gf2*cos(pix)*cos(piy)
c     
            xx = 0.d00
            yy = 0.d00
            zz = 0.d00
            do i=1,nen
               xx = xx + shl(4,i,l)*xl(1,i)
               yy = yy + shl(4,i,l)*xl(2,i)
               zz = zz + shl(4,i,l)*xl(3,i)
            end do
c
            pix = pi*xx
            piy = pi*yy
            piz = pi*zz
            sx = dsin(pix)
            sy = dsin(piy)
            sz = dsin(piz)
            cx = dcos(pix)
            cy = dcos(piy)
            cz = dcos(piz)
c
            pi2 = pi*pi
            co = 1.d00
c
            pss = 3.d00*eps*pi2*sx*sy*sz
c            
c$$$            pss = gf1*(2.d00*eps*pi2*sx*sy+b1*pi*cx*sy+b2*pi*sx*cy)
c$$$     &           + gf2*(-pi*(-dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)*eps
c$$$     &           + dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)*eps*dexp((yy -
c$$$     &           0.1D1)/eps) + dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)
c$$$     &           *eps*dexp((xx - 0.1D1)/eps) - dsin(pix/0.2D1)*pi*dsin(piy/0.2D1)
c$$$     &           *eps*dexp((xx - 0.2D1 + yy)/eps) - dcos(pix/0.2D1
c$$$     &           )*dsin(piy/0.2D1)*dexp((xx - 0.1D1)/eps) + dcos(pix/0.2D1
c$$$     &           )*dsin(piy/0.2D1)*dexp((xx - 0.2D1 + yy)/eps) - dsin(pix/0.2D1
c$$$     &           )*dcos(piy/0.2D1)*dexp((yy - 0.1D1)/eps) + dsin(pix/0.2D1
c$$$     &           )*dcos(piy/0.2D1)*dexp((xx - 0.2D1 + yy)/eps)
c$$$     &           - dcos(pix/0.2D1)*dsin(piy/0.2D1) + dcos(pix/0.2D1)
c$$$     &           *dsin(piy/0.2D1)*dexp((yy - 0.1D1)/eps) - dsin(pix/0.2D1
c$$$     &           )*dcos(piy/0.2D1) + dsin(pix / 0.2D1)*dcos(piy/0.2D1
c$$$     &           )*dexp((xx - 0.1D1)/eps))/0.2D1)
c$$$  &           + gf3*1.d00
c
            
c     **************************************************************************
c     loop to compute volume integrals
c     **************************************************************************     
            do j=1,nenp
               djx = shgp(1,j,l)*c1
               djy = shgp(2,j,l)*c1
               djz = shgp(3,j,l)*c1
               djn = shgp(4,j,l)*c1
c 
c     source terms
c     
               nbj  = ned*(j-1)
               nbj1 = nbj+1
c     
c     (grad p , grad q) + (b.grad p,q) = (f,q)
c     
               elfbb(nbj1) = elfbb(nbj1) + djn*pss
c     
c     element stiffness
c     
               do i=1,nenp
                  nbi  = ned*(i-1)
                  nbi1 = nbi+1
c     
                  dix = shgp(1,i,l)
                  diy = shgp(2,i,l)
                  diz = shgp(3,i,l)
                  din = shgp(4,i,l)
c     
c     (grad p, grad q) + (b.grad p,q) = (f,q)
c     
                  elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                             + eps*(dix*djx + diy*djy + diz*djz)
     &                             + (b1*djx + b2*djy + b3*djz)*din
               end do     
            end do
         end do
         
c     **************************************************************************     
c     boundary terms
c     **************************************************************************
         
         do 4000 ns=1,nside
c     
c     localiza os parametros do lado ns
c     
            do nn=1,npars
               nld = (ns-1)*npars + nn
               dls(nn) = dl(1,nld)
            end do
c     
c     localiza os no's do lado ns
c     
c     
            ns1 = idside(ns,1)
            ns2 = idside(ns,2)
            ns3 = idside(ns,3)
            ns4 = idside(ns,4)

            nl1 = ien(ns1,nel)
            nl2 = ien(ns2,nel)
            nl3 = ien(ns3,nel)
            nl4 = ien(ns4,nel)
c     
c$$$            if(nl2.gt.nl1) then
c$$$               sign = 1.d00
c$$$               do  nn=1,nenlad
c$$$                  idlsd(nn)=idside(ns,nn)
c$$$               end do
c$$$            else
c$$$               sign = -1.d00
c$$$               idlsd(1) = idside(ns,2)
c$$$               idlsd(2) = idside(ns,1)
c$$$               id3 = nenlad-2
c$$$               if(id3.gt.0) then
c$$$                  do il=id3,nenlad
c$$$                     idlsd(il) = idside(ns,nenlad+id3-il)
c$$$                  end do
c$$$               end if
c$$$            end if
c

c     
c     calcula sign para a aresta/face
c
            call vec3sub(xl(1,ns2), xl(1,ns1), vab)
            call vec3sub(xl(1,ns4), xl(1,ns1), vad)
            call vec3sub(coo,xl(1,ns1),vc)
            call cross(vab,vad,xn)
            call nrm3(xn)
            dotcn = dot3(vc,xn)
            if(dotcn.lt.0.d0) then
               sign = 1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
            else
               write(*,'(A)') " trocou sinal"
               sign = -1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
            end if
            write(*,'(A,F8.4)') " sinal", sign
c
c     confere nos globais depois que trocou (ou nao)
c
            ns1=idlsd(1)
            ns2=idlsd(2)
            ns3=idlsd(3)
            ns4=idlsd(4)

            nl1=ien(ns1,nel)
            nl2=ien(ns2,nel)
            nl3=ien(ns3,nel)
            nl4=ien(ns4,nel)
            write(*,'(A,4I5)') " nodes globais", nl1,nl2,nl3,nl4
c
c     ajeita a normal com o sinal
c
            xn(1)=sign*xn(1)
            xn(2)=sign*xn(2)
            xn(3)=sign*xn(3)
c
c     localize the coordinates of the side nodes
c            
            do nn=1,nenlad
               nl = idlsd(nn)
               xls(1,nn) = xl(1,nl)
               xls(2,nn) = xl(2,nl)
               xls(3,nn) = xl(3,nl)
            end do
c
c     global shape functions 
c            
            if(nesq.eq.2) then
               call oneshgp(xls,detn,shlb,shln,shgn,
     &              nenlad,nnods,nints,nesd,ns,nel,neg)
               call oneshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad,npars,nints,nesd,ns,nel,neg)
            else if(nesd.eq.3) then
               nenlad2=4
               npars2=npars
               nnods2=4
ccc               call twoshgp(xls,detn,shlb,shln,shgn,
ccc     &              nenlad2,nnods2,nints,nesd,ns,nel,neg,xxn)
               call twoshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad2,npars2,nints,nesd,ns,nel,neg,xxn)
            end if
c     
c     compute boundary integral
c     
            do 1000 ls=1,nints
               lb = (ns-1)*nints+ls
c     
c     valores dos parametros do multiplicador
c     
               dhs = 0.d00
               do i=1,npars
                  dhs = dhs + dls(i)*shgpn(3,i,ls)
               end do
c     
c     geometria
c     
               x1 = 0.d00
               x2 = 0.d00
               x3 = 0.d00
c               dx1 = 0.d00
c               dx2 = 0.d00
c     
               do i=1,nenlad
                  x1 = x1 + xls(1,i)*shlb(3,i,ls)
                  x2 = x2 + xls(2,i)*shlb(3,i,ls)
                  x3 = x3 + xls(2,i)*shlb(3,i,ls)
c                 dx1 = dx1 + xls(1,i)*shlb(1,i,ls)
c                 dx2 = dx2 + xls(2,i)*shlb(1,i,ls)
               end do
c               dxx=dsqrt(dx1*dx1+dx2*dx2)
c               xn1= sign*dx2/dxx
c               xn2=-sign*dx1/dxx

c     
c     normal da face
c
               xn1 = xn(1)
               xn2 = xn(2)
               xn3 = xn(3)               
c
               bn = b1*xn1 + b2*xn2 + b3*xn3
               en1 = sign*xn1
               en2 = sign*xn2
               en3 = sign*xn3
c     
c     valores exatos dos multiplicadores
c     
               pix = pi*x1
               piy = pi*x2
               piz = pi*x3
               sx = dsin(pix)
               sy = dsin(piy)
               sz = dsin(piz)
               cx = dcos(pix)
               cy = dcos(piy)
               cz = dcos(piz)
c     
               pi2 = pi*pi
               xinvpes = eps**(-1.0)
               co = 1.d00
c     
c     valor exato do multiplicador
c               
c     dhs = gf1*sx*sy
c      &    + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*(1.d00 - dexp((x1
c      &          - 1.d00)*inveps))*(1.d00 - dexp((x2 - 1.d00)*inveps)))
c      &    + gf3*1.d00
c     
               do j=1,nenp
                  nbj  = ned*(j-1)
                  nbj1 = nbj+1
                  djx = shgpsd(1,j,lb)*detpn(ls)*wn(ls)
                  djy = shgpsd(2,j,lb)*detpn(ls)*wn(ls)
                  djz = shgpsd(3,j,lb)*detpn(ls)*wn(ls)
                  djn = shgpsd(4,j,lb)*detpn(ls)*wn(ls)
                  gjn = djx*xn1 + djy*xn2 + djz*xn3
c     
                  elfbb(nbj1) = elfbb(nbj1) 
     &                        + eps*(betah*dhs*djn - dhs*gjn)
     &                        + upwind*dhs*max(0.d00,-bn)*djn
c                  
                  do i=1,nenp
                     nbi  = ned*(i-1)
                     nbi1 = nbi+1
                     dix = shgpsd(1,i,lb)
                     diy = shgpsd(2,i,lb)
                     diz = shgpsd(3,i,lb)
                     din = shgpsd(4,i,lb)
                     gin = dix*xn1 + diy*xn2 + diz*xn3
                     elmbb(nbi1,nbj1) = elmbb(nbi1,nbj1)
     &                                - eps*(gin*djn + din*gjn)
     &                                + eps*betah*din*djn
     &                                + upwind*din*max(0.d00,-bn)*djn
                  end do
               end do
c     
 1000       continue
 4000    continue
c     
c     local TLDG-solution (potencial)
c     
         call solvetdg(elmbb,elfbb,neep)
c     
c     valores nodais descontinuos
c
         do j=1,nenp
            ddis(1,j,nel) = elfbb(j)
         end do     
c     
 500  continue
c     
      return
      end

c-------------------------------------------------------------------------------
      subroutine dualbc(shgpn,shlb,detpn,wn,
     &            gf1,gf2,gf3,xls,elfc,eps,pi,sign,
     &            nints,nenlad,ns,npars)
c-------------------------------------------------------------------------------
c     Dirichlet boundary conditions
c-------------------------------------------------------------------------------
c
      implicit real*8(a-h,o-z)
      dimension shlb(3,nenlad,*),shgpn(3,npars,*),xls(2,*)
      dimension detpn(*),wn(*),elfc(*)
c
      do 1000 ls=1,nints
c
c     geometria
c
        x1 =0.d00
        x2 =0.d00
        dx1=0.d00
        dx2=0.d00
c
        do i=1,nenlad
          x1 =x1 +xls(1,i)*shlb(2,i,ls)
          x2 =x2 +xls(2,i)*shlb(2,i,ls)
          dx1=dx1+xls(1,i)*shlb(1,i,ls)
          dx2=dx2+xls(2,i)*shlb(1,i,ls)
        end do
           dxx=dsqrt(dx1*dx1+dx2*dx2)
           xn1= sign*dx2/dxx
           xn2=-sign*dx1/dxx
c
c
      pix=pi*x1
      piy=pi*x2
      sx=dsin(pix)
      sy=dsin(piy)
      cx=dcos(pix)
      cy=dcos(piy)
c
      pi2=pi*pi
      xinvpes = eps**(-1.0)
      co=1.d00
c
c    valor exato do multiplicador
c
      dhse =  gf1*sx*sy
     &    + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*(1.d00 - dexp((x1
     &          - 1.d00)/eps))*(1.d00 - dexp((x2 - 1.d00)/eps)))
     &    + gf3*1.d00
c
c
      dhsx= gf1*pi*cx*sy
     &    + gf2*(dcos(pix/0.2D1)*pi*dsin(piy/0.2D1)*(0.1D1 - dexp((x1
     & - 0.1D1)/eps))*(0.1D1 - dexp((x2 - 0.1D1)/eps))/0.2D1 -
     & dsin(pix/0.2D1)*dsin(piy/0.2D1)/eps*dexp((x1 - 0.1D1)/eps)*(0.1D1
     &  - dexp((x2 - 0.1D1)/eps)))
     &   +gf3*1.d00
c
      dhsy= gf1*pi*sx*cy
     &    - gf2*(dsin(pix/0.2D1)*dcos(piy/0.2D1)*pi*(0.1D1 - dexp((x1
     & - 0.1D1)/eps))*(0.1D1 - dexp((x2 - 0.1D1)/eps))/0.2D1 -
     & dsin(pix/0.2D1)*dsin(piy/0.2D1)*(0.1D1 - dexp((x1 - 0.1D1)/eps
     & ))/eps*dexp((x2 - 0.1D1)/eps))
     &   +gf3*1.d00
c
       gradn=(dhsx*xn1+dhsy*xn2)
c
        do 1100 j=1,npars
c
          ncj1 = (ns-1)*npars + j
c
         djn=shgpn(2,j,ls)*detpn(ls)*wn(ls)
c
c    source term
c
         elfc(ncj1) = elfc(ncj1) + gradn*djn
c
 1100 continue
 1000 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine drchbc(shgpn,shlb,detpn,wn,gf1,gf2,gf3,xls,dls,eps,pi,
     &                  nints,nenlad,npars)
c-----------------------------------------------------------------------
c     Dirichlet boundary conditions
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      dimension shlb(3,nenlad,*),shgpn(3,npars,*),xls(2,*)
      dimension detpn(*),wn(*),dls(*)
      dimension aa(10,10),bb(10)
c
      do i=1,npars
	    bb(i) = 0.d00
         do j=1,npars
            aa(i,j) = 0.d00
         end do
      end do
c
      do 1000 ls=1,nints
c
c     geometria
c
         x1 = 0.d00
         x2 = 0.d00
c
         do i=1,nenlad
            x1 = x1 +xls(1,i)*shlb(2,i,ls)
            x2 = x2 +xls(2,i)*shlb(2,i,ls)
         end do
c
         pix = pi*x1
         piy = pi*x2
         sx = dsin(pix)
         sy = dsin(piy)
         cx = dcos(pix)
         cy = dcos(piy)
c
         pi2 = pi*pi
         xinvpes = eps**(-1.0)
         co = 1.d00
c
c     valor exato do multiplicador
c
         dhse = gf1*sx*sy
     &        + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*(1.d00 - dexp((x1
     &        - 1.d00)/eps))*(1.d00 - dexp((x2 - 1.d00)/eps)))
     &        + gf3*1.d00
c
         do 1100 j=1,npars
c
c
            djn=shgpn(2,j,ls)*detpn(ls)*wn(ls)
c
c     source term
c
            bb(j) = bb(j) + dhse*djn
c
            do i=1,npars
c
               din=shgpn(2,i,ls)
c
c     L2-projection at the boundary (side)
c
               aa(i,j) = aa(i,j) + din*djn
c
            end do
c
 1100    continue
 1000 continue
c
      call invmb(aa,10,npars)
c
      do i=1,npars
         dls(i) = 0.d00
         do j=1,npars
            dls(i) = dls(i) + aa(i,j)*bb(j)
         end do
      end do
      return
      end

c-----------------------------------------------------------------------
      subroutine drchbc3(shgpn,shlb,detpn,wn,gf1,gf2,gf3,xls,dls,eps,pi,
     &                   nints,nenlad,npars)
c-----------------------------------------------------------------------
c     Dirichlet boundary conditions
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
c
      dimension shlb(3,nenlad,*),shgpn(3,npars,*),xls(3,*)
      dimension detpn(*),wn(*),dls(*)
      dimension aa(10,10),bb(10)
c
      do i=1,npars
	    bb(i) = 0.d00
         do j=1,npars
            aa(i,j) = 0.d00
         end do
      end do
c
c      write(*,*) "NPARS=",npars
c      write(*,*) "NENLAD=",nenlad
c      do i=1,nenlad
c         write(*,'(10F8.4)') (xls(k,i),k=1,3)
c     end do

c
      do ls=1,nints
c
c     geometria
c
         x1 = 0.d00
         x2 = 0.d00
         x3 = 0.d00
c
         do i=1,nenlad
            x1 = x1 + xls(1,i)*shlb(3,i,ls)
            x2 = x2 + xls(2,i)*shlb(3,i,ls)
            x3 = x3 + xls(3,i)*shlb(3,i,ls)
         end do
c
         pix=pi*x1
         piy=pi*x2
         piz=pi*x3
         sx=dsin(pix)
         sy=dsin(piy)
         sz=dsin(piz)
         cx=dcos(pix)
         cy=dcos(piy)
         cz=dcos(piz)
c
         pi2=pi*pi
         xinvpes = eps**(-1.0)
         co=1.d00
c
c     valor exato do multiplicador
c
         dhse = gf1*sx*sy*sz
     &        + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*(1.d00 - dexp((x1
     &        - 1.d00)/eps))*(1.d00 - dexp((x2 - 1.d00)/eps)))
     &        + gf3*1.d00
c
         do j=1,npars
            djn = shgpn(3,j,ls)*detpn(ls)*wn(ls)
c
c     source term
c
            bb(j) = bb(j) + dhse*djn
c
            do i=1,npars
               din = shgpn(3,i,ls)
c
c     L2-projection at the boundary (side)
c
               aa(i,j) = aa(i,j) + din*djn
            end do
c
         end do
      end do
c
      call invmb(aa,10,npars)
c
      do i=1,npars
         dls(i) = 0.d00
         do j=1,npars
            dls(i) = dls(i) + aa(i,j)*bb(j)
         end do
      end do
      return
      end

c-----------------------------------------------------------------------
      subroutine solvetdg(elmbb,elfbb,neep)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension elmbb(neep,*),elfbb(*)
      dimension fab(200)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
c     resolve o sistema
c
c     B Xb = Fb
c
      call invmb(elmbb,neep,neep)
c
c     Aa = A^{-1} Fa
c
      do j=1,neep
         fab(j)=0.d00
         do k=1,neep
            fab(j) = fab(j) + elmbb(j,k)*elfbb(k)
         end do
      end do
c
c     xb=fab
c
      do i=1,neep
         elfbb(i) = fab(i)
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine solvedh(elma,elmb,elmd,elmh,
     &     elfa,elfb,elfd,elfab,necon,neep)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension elma(necon,*),elmb(necon,*),elmd(neep,*),
     &          elmh(neep,*)
      dimension elfa(*),elfb(*),elfd(*),elfab(*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
c   resolve o sistema
c
c   A Xa + B Xb = Fa
c
c   B^t Xa + H Xb = Fb
c
c
      call invmb(elma,necon,necon)
c
c    Aa = A^{-1} Fa
c
      do j=1,necon
        elfab(j)=0.d00
        do k=1,necon
          elfab(j) = elfab(j) + elma(j,k)*elfa(k)
        end do
      end do
c
c  D = (B^T) (A^{-1})
c
      do i=1,neep
      do j=1,necon
        elmd(i,j)=0.d00
        do k=1,necon
          elmd(i,j) = elmd(i,j) + elmb(k,i)*elma(k,j)
        end do
      end do
      end do
c
c  H = H - B^T A^{-1}B
c
      do i=1,neep
      do j=1,neep
        do k=1,necon
          elmh(i,j) = elmh(i,j) - elmd(i,k)*elmb(k,j)
        end do
      end do
      end do
c
c    Fd = Fb - B^T A^{-1} Fa = Fb - D Fa
c
      do j=1,neep
        elfd(j)=elfb(j)
        do k=1,necon
          elfd(j) = elfd(j) - elmd(j,k)*elfa(k)
        end do
      end do
c
      call invmb(elmh,neep,neep)
c
c    Calcula Xb
c
      do j=1,neep
        elfb(j)=0.d00
        do k=1,neep
          elfb(j) = elfb(j) + elmh(j,k)*elfd(k)
        end do
      end do
c
c   Calcula Xa
c
      do j=1,necon
        elfa(j)=elfab(j)
        do k=1,neep
          elfa(j) = elfa(j) - elmd(k,j)*elfb(k)
        end do
      end do
c
      return
c
      end

c-----------------------------------------------------------------------
      subroutine invmb(am,ndim,m)
c-----------------------------------------------------------------------
c     subrotina de inversao
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension ipi(200),ind(200,2),piv(200),dis(200,1),am(ndim,*)
      ncoln=0
      det=1.0
      do 20 j=1,m
   20 ipi(j)=0
      do 550 i=1,m
      amax=0.0
      do 105 j=1,m
      if(ipi(j)-1)60,105,60
   60 do 100 k=1,m
      if(ipi(k)-1) 80,100,740
   80 if(dabs(amax)-dabs(am(j,k)))85,100,100
   85 irow=j
      ico=k
      amax=am(j,k)
  100 continue
  105 continue
      ipi(ico)=ipi(ico)+1
      if(irow-ico)140,260,140
  140 det=-det
      do 200 l=1,m
      swap=am(irow,l)
      am(irow,l)=am(ico,l)
  200 am(ico,l)=swap
      if(ncoln) 260,260,210
  210 do 250 l=1,ncoln
      swap=dis(irow,l)
      dis(irow,l)=dis(ico,l)
  250 dis(ico,l)=swap
  260 ind(i,1)=irow
      piv(i)=am(ico,ico)
      ind(i,2)=ico
      det=det*piv(i)
      am(ico,ico)=1.0
      do 350 l=1,m
  350 am(ico,l)=am(ico,l)/piv(i)
      if(ncoln) 380,380,360
  360 do 370 l=1,ncoln
  370 dis(ico,l)=dis(ico,l)/piv(i)
  380 do 550 lz=1,m
      if(lz-ico)400,550,400
  400 t=am(lz,ico)
      am(lz,ico)=0.0
      do 450 l=1,m
  450 am(lz,l)=am(lz,l)-am(ico,l)*t
      if(ncoln)550,550,460
  460 do 500 l=1,ncoln
  500 dis(lz,l)=dis(lz,l)-dis(ico,l)*t
  550 continue
      do 710 i=1,m
      l=m+1-i
      if(ind(l,1)-ind(l,2))630,710,630
  630 jrow=ind(l,1)
      jco=ind(l,2)
      do 705 k=1,m
      swap=am(k,jrow)
      am(k,jrow)=am(k,jco)
      am(k,jco)=swap
  705 continue
  710 continue
  740 continue
      return
      end

c-----------------------------------------------------------------------
      subroutine fluxmx(c,numat)
c-----------------------------------------------------------------------
c     program to read, write and store properties
c     for plane stres mixed elements
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension c(10,*)
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      do 100 n=1,numat
      if (mod(n,50).eq.1) write(ieco,1000) numat
c
      read(iin,2000) m,del1,del2,del3,del4,
     &                 del5,del6,del7,del8,del9
      c(1,m)=del1! eps
      c(2,m)=del2! b1
      c(3,m)=del3! b2
      c(4,m)=del4! apwind (ativa o termo upwind, ver Oikawa 2014)
      c(5,m)=del5
      c(6,m)=del6
	c(7,m)=del7
	c(8,m)=del8
	c(9,m)=del9
      write(ieco,3000) m,del1,del2,del3,del4,
     &                    del5,del6,del7,del8,del9
c

  100 continue
c
      return
c
 1000 format(///,
     &' m a t e r i a l   s e t   d a t a       '   //5x,
     &' number of material sets . . . . . (numat ) = ',i10//,
     & 7x,'set',7x,'eps',7x,'vel1',
     & 7x,'vel2',7x,'del4',7x,'del5',7x,'del6',7x,'del7',7x,'del8',/)
 2000 format(i10,10f10.0)
 3000 format(i10,2x,8(1x,1pe10.3))
      end

c-----------------------------------------------------------------------
      subroutine uexafx(x,y,ue,duex,duey,eps,index)
c-----------------------------------------------------------------------
c     solucao exata e derivada
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension ue(*),duex(*),duey(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      go to (100,200,300) index
c
c     PROBLEM 1 - T=sin\B9x*sin\B9y
c
 100  continue
      pi=4.d00*datan(1.d00)
      pi2=pi*pi
      px=pi*x
      py=pi*y
      sx=dsin(px)
      sy=dsin(py)
      cx=dcos(px)
      cy=dcos(py)
      co=1.d00
c
      ue(1) = -pi*cx*sy
      ue(2) = -pi*sx*cy
      ue(3) = sx*sy
c
      duex(1) =  pi2*sx*sy
      duex(2) = -pi2*cx*cy
      duex(3) =  pi*cx*sy
c
      duey(1) = -pi2*cx*cy
      duey(2) =  pi2*sx*sy
      duey(3) =  pi*sx*cy
      go to 1001
c
c     2  PROBLEM 2 - U= sin(pix/2)sin(piy/2)*(1-e^((x-1)/eps))
c     &                 *(1-e^((x-1)/eps))
c
 200  continue
      pi=4.d00*datan(1.d00)
      pi2=pi*pi
      px=pi*x
      py=pi*y
      sx=dsin(px)
      sy=dsin(py)
      cx=dcos(px)
      cy=dcos(py)
      xinveps=eps**(-1.0)
      co=1.d00
c
      ue(1) = -(pi*dcos(px/2.d00)*dsin(py/2.d00)*(1.d00 - dexp((x
     &  - 1.d00)*xinveps))*(1.d00 - dexp((y - 1.d00)*xinveps))/2.d00-
     &  dsin(px/2.d00)*dsin(py/2.d00)*dexp((x - 1.d00)*xinveps)*(1.d00
     & -dexp((y - 1.d00)*xinveps))*xinveps)
c
      ue(2) =  -(pi*dsin(px/2.d00)*dcos(py/2.d00)*(1.d00 - dexp((x
     &     - 1.d00)/eps))*(1.d00 - dexp((y - 1.d00)/eps))/2.d00 -
     &     dsin(px/2.d00)*dsin(py/2.d00)*(1.d00 - dexp((x - 1.d00)/eps
     &     ))*dexp((y - 1.d00)/eps)/eps)
c
      ue(3) = dsin(px/2.d00)*dsin(py/2.d00)*(1.d00 - dexp((x
     &     - 1.d00)/eps))*(1.d00 - dexp((y - 1.d00)/eps))
c
      duex(1)= -(-dsin(py/2.d00)*(-1.d00 + dexp((y - 1.d00) /eps))*(
     &     -dsin(px/2.d00)*pi**2*eps**2 + dsin(px/2.d00)*
     &     pi**2*eps**2*dexp((x - 1.d00)/eps) - 4.d00*dcos(px/2.d00
     &     )*pi*dexp((x - 1.d00)/eps)*eps - 4.d00*dsin(px/2.d00
     &     )*dexp((x - 1.d00)/eps))/eps**2/4.d00)
c
      duex(2)= -((dcos(px/2.d00)*pi**2*dcos(py/2.d00)*eps**2
     &     - dcos(px/2.d00)*pi**2*dcos(py/2.d00)*eps**2
     &     *dexp((y - 0.1D1)/eps) - dcos(px/0.2D1)*pi**2*dcos(
     &     py/0.2D1)*eps**2*dexp((x - 0.1D1)/eps) + dcos(px/ 0.2D1)
     &     *pi**2*dcos(py/0.2D1)*eps**2*dexp((x - 0.2D1
     &     + y)/eps) - 0.2D1*dcos(px/0.2D1)*pi*dsin(py/0.2D1)
     &     *dexp((y - 0.1D1)/eps)*eps + 0.2D1*dcos(px/0.2D1)
     &     *pi*dsin(py/0.2D1)*dexp((x - 0.2D1 + y)/eps)*eps - 0.2D1
     &     *dsin(px/0.2D1)*dcos(py/0.2D1)*pi*dexp((x - 0.1D1
     &     )/eps)*eps + 0.2D1*dsin(px/0.2D1)*dcos(py/0.2D1
     &     )*pi*dexp((x - 0.2D1 + y)/eps)*eps + 0.4D1*dsin(px/0.2D1
     &     )*dsin(py/0.2D1)*dexp((x - 0.2D1 + y)/eps))/eps**2/0.4D1)
c
      duex(3)= pi*dcos(px/2.d00)*dsin(py/2.d00)*(1.d00 - dexp((x
     &     - 1.d00)/eps))*(1.d00 - dexp((y - 1.d00)/eps))/2.d00 -
     &     dsin(px/2.d00)*dsin(py/2.d00)/eps*dexp((x-1.d00)/eps)*(1.d00
     &     - dexp((y - 1.d00)/eps))
c
      duey(1)= -(-dsin(px/0.2D1)*(-0.1D1 + dexp((x - 0.1D1)/eps))*(
     &     -pi**2*dsin(py/0.2D1)*eps**2 + pi**2*dsin(py/0.2D1
     &     )*eps**2*dexp((y - 0.1D1)/eps) - 0.4D1*dcos(py/0.2D1
     &     )*pi*dexp((y - 0.1D1)/eps)*eps - 0.4D1*dsin(py/0.2D1
     &     )*dexp((y - 0.1D1)/eps))/eps**2/0.4D1)
c
      duey(2)= -((dcos(px/0.2D1)*pi**2*dcos(py/0.2D1)*eps**2
     &     - dcos(px/0.2D1)*pi**2*dcos(py/0.2D1)*eps**2
     &     *dexp((y - 0.1D1)/eps) - dcos(px/0.2D1)*pi**2*dcos(
     &     py/0.2D1)*eps**2*dexp((x - 0.1D1)/eps) + dcos(px/0.2D1
     &     )*pi**2*dcos(py/0.2D1)*eps**2*dexp((x - 0.2D1
     &     + y)/eps) - 0.2D1*dcos(px/0.2D1)*pi*dsin(py/0.2D1
     &     )*dexp((y - 0.1D1)/eps)*eps + 0.2D1*dcos(px/0.2D1)
     &     *pi*dsin(py/0.2D1)*dexp((x - 0.2D1 + y)/eps)*eps - 0.2D1
     &     *dsin(px/0.2D1)*dcos(py/0.2D1)*pi*dexp((x - 0.1D1
     &     )/eps)*eps + 0.2D1*dsin(px/0.2D1)*dcos(py/0.2D1
     &     )*pi*dexp((x - 0.2D1 + y)/eps)*eps + 0.4D1*dsin(px/0.2D1
     &     )*dsin(py/0.2D1)*dexp((x - 0.2D1 + y)/eps))/eps**2/0.4D1)
c
      duey(3)=dsin(px/0.2D1)*dcos(py/0.2D1)*pi*(0.1D1 - dexp((x
     &     - 0.1D1)/eps))*(0.1D1 - dexp((y - 0.1D1)/eps))/0.2D1 -
     &     dsin(px/0.2D1)*dsin(py/0.2D1)*(0.1D1 - dexp((x - 0.1D1)/eps
     &     ))/eps*dexp((y - 0.1D1)/eps)
c
      go to 1001
c
c     Polinomial
c
 300  continue
c
      ue(1) = 1.d00
      ue(2) = 1.d00
      ue(3) = 1.d00
c
      duex(1)= 1.d00
      duex(2)= 1.d00
      duex(3)=1.d00
c
      duey(1)= 1.d00
      duey(2)= 1.d00
      duey(3)=1.d00
c
 1001 continue
      return
      end

c-----------------------------------------------------------------------
      subroutine uexafx3(x,y,z,ue,duex,duey,duez,eps,index)
c-----------------------------------------------------------------------
c     solucao exata e derivada
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension ue(*),duex(*),duey(*),duez(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      go to (100) index
c
c     PROBLEM 3D: T = sin(pix)sin(piy)sin(piz)
c
 100  continue
      pi=4.d00*datan(1.d00)
      pi2=pi*pi
      px=pi*x
      py=pi*y
      pz=pi*z
      sx=dsin(px)
      sy=dsin(py)
      sz=dsin(pz)
      cx=dcos(px)
      cy=dcos(py)
      cz=dcos(pz)
      co=1.d00
c
c     ue(1) e ue(2) sao para darcy
c     ue(3) a unica que interessa aqui por enquanto
c     similar para duex, duey e duez
c
      ue(1) = -pi*cx*sy
      ue(2) = -pi*sx*cy
      ue(3) = sx*sy*sz
c
      duex(1) = 0.d0
      duex(2) = 0.d0
      duex(3) = pi*cx*sy*sz
c
      duey(1) = 0.d0
      duey(2) = 0.d0
      duey(3) = pi*sx*cy*sz
c
      duez(1) = 0.d0
      duez(2) = 0.d0
      duez(3) = pi*sx*sy*cz
c
      go to 1001
c
 1001 continue
c
      return
      end

c-----------------------------------------------------
      subroutine elemdlf(xl,dlf,eps,ncon,nencon,index)
c-----------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension xl(2,*),dlf(ncon,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      pi=4.d00*datan(1.d00)
      pi2=pi*pi
c
      do 1001 n=1,nencon
         x = xl(1,n)
         y = xl(2,n)
         go to (100,200,300) index
c
c     PROBLEM 1 - T=sin\B9x*sin\B9y
c
 100     continue
         px=pi*x
         py=pi*y
         sx=dsin(px)
         sy=dsin(py)
         cx=dcos(px)
         cy=dcos(py)
         co=1.d00
c
         dlf(1,n) = -pi*cx*sy
         dlf(2,n) = -pi*sx*cy
c
         go to 1001
c
c     2  PROBLEM 2 - U=  sin(pix/2)sin(piy/2)*(1-e^((x-1)/eps))
c     &                 *(1-e^((x-1)/eps))
c
 200     continue
         px=pi*x
         py=pi*y
         sx=dsin(px)
         sy=dsin(py)
         cx=dcos(px)
         cy=dcos(py)
         co=1.d00
c
      dlf(1,n) = -(dcos(pi*x/0.2D1)*pi*dsin(pi*y/0.2D1)*(0.1D1 - dexp((x
     &  - 0.1D1)/eps))*(0.1D1 - dexp((y - 0.1D1)/eps))/0.2D1 -
     & dsin(pi*x/0.2D1)*dsin(pi*y/0.2D1)/eps*dexp((x -0.1D1)/eps)*(0.1D1
     &    - dexp((y - 0.1D1)/eps)))
c
      dlf(2,n) =  -(dsin(pi*x/0.2D1)*dcos(pi*y/0.2D1)*pi*(0.1D1 -dexp((x
     &        - 0.1D1)/eps))*(0.1D1 - dexp((y - 0.1D1)/eps))/0.2D1 -
     &   dsin(pi*x/0.2D1)*dsin(pi*y/0.2D1)*(0.1D1 - dexp((x - 0.1D1)/eps
     &        ))/eps*dexp((y - 0.1D1)/eps))
c
         go to 1001
c
c     Polinomial
c
 300     continue
c
         dlf(1,n) = 1.d00
         dlf(2,n) = 1.d00
c
 1001 continue
      return
      end

c-----------------------------------------------------
      subroutine elemdlp(xl,dlp,eps,ned,nenp,index)
c-----------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension xl(2,*),dlp(ned,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      pi=4.d00*datan(1.d00)
      pi2=pi*pi
c
      do 1001 n=1,nenp
       x = xl(1,n)
       y = xl(2,n)
       go to (100,200,300) index
c
c        PROBLEM 1 - T=sin\B9x*sin\B9y
c
 100   continue
      px=pi*x
      py=pi*y
      sx=dsin(px)
      sy=dsin(py)
      cx=dcos(px)
      cy=dcos(py)
      co=1.d00
c
      dlp(1,n) = sx*sy
c
      go to 1001
c
c     2  PROBLEM 2 - U=  sin(pix/2)sin(piy/2)*(1-e^((x-1)/eps))
c    &                 *(1-e^((x-1)/eps))
c
 200   continue
      px=pi*x
      py=pi*y
      sx=dsin(px)
      sy=dsin(py)
      cx=dcos(px)
      cy=dcos(py)
      xinveps=eps**(-1.0)
      co=1.d00
c
      dlp(1,n) = dsin(px/2.d00)*dsin(py/2.d00)*(1.d00 - dexp((x
     &- 1.d00)/eps))*(1.d00 - dexp((y - 1.d00)/eps))
c
      go to 1001
c
c     Plane wave
c
300   continue
c
      dlp(1,n) = 1.d00
c
1001  continue
      return
      end
c-----------------------------------------------------------------------
      subroutine elemult(xl,dlp,eps,ned,nenp,index)
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      dimension xl(2,*),dlp(ned,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      pi=4.d00*datan(1.d00)
      pi2=pi*pi
c
      do 1001 n=1,nenp
       x = xl(1,n)
       y = xl(2,n)
c
       go to (100,200,300) index
c
c        PROBLEM 1 - T=sin\B9x*sin\B9y
c
 100   continue
      px=pi*x
      py=pi*y
      sx=dsin(px)
      sy=dsin(py)
      cx=dcos(px)
      cy=dcos(py)
      co=1.d00
c
      dlp(1,n) = sx*sy
c
      go to 1001
c
c     2  PROBLEM 2 - U=  sin(pix/2)sin(piy/2)*(1-e^((x-1)/eps))
c    &                 *(1-e^((x-1)/eps))
c
 200   continue
      px=pi*x
      py=pi*y
      sx=dsin(px)
      sy=dsin(py)
      cx=dcos(px)
      cy=dcos(py)
      xinveps=eps**(-1.0)
      co=1.d00
c
      dlp(1,n) = dsin(px/2.d00)*dsin(py/2.d00)*(1.d00 - exp((x
     &- 1.d00)/eps))*(1.d00 - exp((y - 1.d00)/eps))
c
      go to 1001
c
c     Plane wave
c
300   continue
c
      dlp(1,n) = 1.d00
c
1001  continue
      return
      end
c-----------------------------------------------------------------------
      subroutine shlq(shl,w,nint,nen)
c
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a four-node quadrilateral element
c
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c
      implicit real*8 (a-h,o-z)
c
c     remove above card for single precision operation
c
      dimension wone(8),raone(8)
      dimension shl(3,nen,*),w(*),ra(64),sa(64)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      if (nint.eq.1) then
         wone(1)  = two
         raone(1) = zero
	nintx=1
	ninty=1
      endif
c
c
      if (nint.eq.4) then
         wone(1) = one
         wone(2) = one
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
	nintx=2
	ninty=2
      endif
c
c
      if (nint.eq.9) then
         wone(1) = five9
         wone(2) = five9
         wone(3) = eight9
         raone(1)=-.774596669241483
         raone(2)= .774596669241483
         raone(3)= zero
	nintx=3
	ninty=3
      endif
c
      if (nint.eq.16) then
         wone(1) = .347854845137454
         wone(2) = .347854845137454
         wone(3) = .652145154862546
         wone(4) = .652145154862546
         raone(1)=-.861136311594053
         raone(2)= .861136311594053
         raone(3)=-.339981043584856
         raone(4)= .339981043584856
	nintx=4
	ninty=4
      endif
c
c
       if(nint.eq.25) then
        wone(1) = .236926885056189
        wone(2) = .236926885056189
        wone(3) = .478628670499366
        wone(4) = .478628670499366
        wone(5) = .568888888888888
        raone(1)=-.906179845938664
        raone(2)= .906179845938664
        raone(3)=-.538469310105683
        raone(4)= .538469310105683
        raone(5)= zero
	nintx=5
	ninty=5
       endif
c
       if(nint.eq.36) then
         wone(1) = .171324492397170
         wone(2) = .171324492397170
         wone(3) = .360761573048139
         wone(4) = .360761573048139
         wone(5) = .467913934572691
         wone(6) = .467913934572691
         raone(1)=-.932469514203152
         raone(2)= .932469514203152
         raone(3)=-.661209386466265
         raone(4)= .661209386466365
         raone(5)=-.238619186083197
         raone(6)= .238619186083197
	nintx=6
	ninty=6
        endif
c
c
       if(nint.eq.49) then
         wone(1) = .129484966168870
         wone(2) = .129484966168870
         wone(3) = .279705391489277
         wone(4) = .279705391489277
         wone(5) = .381830050505119
         wone(6) = .381830050505119
         wone(7) = .417959183673469
         raone(1)=-.949107912342759
         raone(2)= .949107912342759
         raone(3)=-.741531185599394
         raone(4)= .741531185599394
         raone(5)=-.405845151377397
         raone(6)= .405845151377397
         raone(7)= zero
	nintx=7
	ninty=7
        endif
c
c
       if(nint.eq.64) then
         wone(1) = .101228536290376
         wone(2) = .101228536290376
         wone(3) = .222381034453374
         wone(4) = .222381034453374
         wone(5) = .313706645877887
         wone(6) = .313706645877887
         wone(7) = .362683783378362
         wone(8) = .362683783378362
         raone(1)=-.960289856497536
         raone(2)= .960289856497536
         raone(3)=-.796666477413627
         raone(4)= .796666477413627
         raone(5)=-.525532409916329
         raone(6)= .525532409916329
         raone(7)=-.183434642495650
         raone(8)= .183434642495650
	nintx=8
	ninty=8
        endif
c
c
         l=0
	   do ly=1,ninty
	   do lx=1,nintx
	   l = l+1
	      w(l) = wone(lx)*wone(ly)
	      ra(l) = raone(lx)
	      sa(l) = raone(ly)
         end do
	   end do
c
      do 200 l=1,nint
c
            r=ra(l)
            s=sa(l)
c
	if(nen.eq.4) then
            f1 = pt5*(one-r)
            f2 = pt5*(one+r)
	      fx1=-pt5
	      fx2= pt5
            g1 = pt5*(one-s)
            g2 = pt5*(one+s)
	      gx1=-pt5
	      gx2= pt5
            shl(1,1,l)=fx1*g1
            shl(2,1,l)=f1*gx1
            shl(3,1,l)=f1*g1
            shl(1,2,l)=fx2*g1
            shl(2,2,l)=f2*gx1
            shl(3,2,l)=f2*g1
            shl(1,3,l)=fx2*g2
            shl(2,3,l)=f2*gx2
            shl(3,3,l)=f2*g2
            shl(1,4,l)=fx1*g2
            shl(2,4,l)=f1*gx2
            shl(3,4,l)=f1*g2
	   end if
c
c
         if(nen.eq.9) then
            f1 = -pt5*(one-r)*r
            f2 =  pt5*(one+r)*r
	      f3 = (one+r)*(one-r)
c
c
	      f1x = pt5*r - pt5*(one-r)
	      f2x = pt5*r + pt5*(one+r)
	      f3x =-two*r

            g1 =-pt5*(one-s)*s
            g2 = pt5*(one+s)*s
	      g3 = (one+s)*(one-s)
c
	      g1x = pt5*s - pt5*(one-s)
	      g2x = pt5*s + pt5*(one+s)
	      g3x =-two*s
c
c
            shl(3,1,l)=f1*g1
            shl(3,2,l)=f2*g1
            shl(3,3,l)=f2*g2
            shl(3,4,l)=f1*g2
c
                  shl(3,5,l)=f3*g1
                  shl(3,6,l)=f2*g3
                  shl(3,7,l)=f3*g2
                  shl(3,8,l)=f1*g3
                  shl(3,9,l)=f3*g3
c
            shl(1,1,l)=f1x*g1
            shl(1,2,l)=f2x*g1
            shl(1,3,l)=f2x*g2
            shl(1,4,l)=f1x*g2
c
                  shl(1,5,l)=f3x*g1
                  shl(1,6,l)=f2x*g3
                  shl(1,7,l)=f3x*g2
                  shl(1,8,l)=f1x*g3
                  shl(1,9,l)=f3x*g3
c
            shl(2,1,l)=f1*g1x
            shl(2,2,l)=f2*g1x
            shl(2,3,l)=f2*g2x
            shl(2,4,l)=f1*g2x
c
                  shl(2,5,l)=f3*g1x
                  shl(2,6,l)=f2*g3x
                  shl(2,7,l)=f3*g2x
                  shl(2,8,l)=f1*g3x
                  shl(2,9,l)=f3*g3x
c
            end if
c
c
            if (nen.eq.16) then
                  onemrsq=one-r*r
                  onemssq=one-s*s
                  onep3r=one+three*r
                  onem3r=one-three*r
                  onep3s=one+three*s
                  onem3s=one-three*s
		  f1=-1.d00/16.d00*(9.d00*r*r-1.d00)*(r-1.d00)
		  f2=9.d00/16.d00*(1.d00-r*r)*onem3r
		  f3=9.d00/16.d00*(1.d00-r*r)*onep3r
		  f4=1.d00/16.d00*(9.d00*r*r-1.d00)*(r+1.d00)
c
		  f1x=-1.d00/16.d00*(18.d00*r)*(r-1.d00)
     &               -1.d00/16.d00*(9.d00*r*r-1.d00)
		  f2x=9.d00/16.d00*(-2.d00*r)*onem3r
     &               -9.d00/16.d00*(1.d00-r*r)*3.d00
		  f3x=9.d00/16.d00*(-2.d00*r)*onep3r
     &               -9.d00/16.d00*(r*r-1.d00)*3.d00
      	  f4x=1.d00/16.d00*(18.d00*r)*(r+1.d00)
     &       +1.d00/16.d00*(9.d00*r*r-1.d00)
c
            g1=-1.d00/16.d00*(9.d00*s*s-1.d00)*(s-1.d00)
		  g2=9.d00/16.d00*(1.d00-s*s)*onem3s
		  g3=9.d00/16.d00*(1.d00-s*s)*onep3s
		  g4=1.d00/16.d00*(9.d00*s*s-1.d00)*(s+1.d00)
c
		  g1x=-1.d00/16.d00*(18.d00*s)*(s-1.d00)
     &               -1.d00/16.d00*(9.d00*s*s-1.d00)
		  g2x=9.d00/16.d00*(-2.d00*s)*onem3s
     &               -9.d00/16.d00*(1.d00-s*s)*3.d00
		  g3x=9.d00/16.d00*(-2.d00*s)*onep3s
     &               -9.d00/16.d00*(s*s-1.d00)*3.d00
            g4x=1.d00/16.d00*(18.d00*s)*(s+1.d00)
     &         +1.d00/16.d00*(9.d00*s*s-1.d00)
c
           shl(3,1,l)=f1*g1
	     shl(3,2,l)=f4*g1
	     shl(3,3,l)=f4*g4
	     shl(3,4,l)=f1*g4
c
           shl(3,5,l)=f2*g1
	     shl(3,6,l)=f3*g1
	     shl(3,7,l)=f4*g2
	     shl(3,8,l)=f4*g3
	     shl(3,9,l)=f3*g4
	     shl(3,10,l)=f2*g4
	     shl(3,11,l)=f1*g3
	     shl(3,12,l)=f1*g2
	     shl(3,13,l)=f2*g2
	     shl(3,14,l)=f3*g2
	     shl(3,15,l)=f3*g3
	     shl(3,16,l)=f2*g3
c
c
           shl(1,1,l)=f1x*g1
	     shl(1,2,l)=f4x*g1
	     shl(1,3,l)=f4x*g4
	     shl(1,4,l)=f1x*g4
c
           shl(1,5,l)=f2x*g1
	     shl(1,6,l)=f3x*g1
	     shl(1,7,l)=f4x*g2
	     shl(1,8,l)=f4x*g3
	     shl(1,9,l)=f3x*g4
	     shl(1,10,l)=f2x*g4
	     shl(1,11,l)=f1x*g3
	     shl(1,12,l)=f1x*g2
	     shl(1,13,l)=f2x*g2
	     shl(1,14,l)=f3x*g2
	     shl(1,15,l)=f3x*g3
	     shl(1,16,l)=f2x*g3
c
c
           shl(2,1,l)=f1*g1x
	     shl(2,2,l)=f4*g1x
	     shl(2,3,l)=f4*g4x
	     shl(2,4,l)=f1*g4x
c
           shl(2,5,l)=f2*g1x
	     shl(2,6,l)=f3*g1x
	     shl(2,7,l)=f4*g2x
	     shl(2,8,l)=f4*g3x
	     shl(2,9,l)=f3*g4x
	     shl(2,10,l)=f2*g4x
	     shl(2,11,l)=f1*g3x
	     shl(2,12,l)=f1*g2x
	     shl(2,13,l)=f2*g2x
	     shl(2,14,l)=f3*g2x
	     shl(2,15,l)=f3*g3x
	     shl(2,16,l)=f2*g3x
c
            end if
  200 continue
c
      return
      end

c-----------------------------------------------------------------------
      subroutine legdre1d(shlone,wone,nint,nen)
c
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a four-node quadrilateral element
c
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c
      implicit real*8 (a-h,o-z)
c
c     remove above card for single precision operation
c
      dimension wone(*),raone(8)
	dimension shlone(3,nen,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      if (nint.eq.1) then
         wone(1)  = two
         raone(1) = zero
	nintx=1
	ninty=1
      endif
c
c
      if (nint.eq.2) then
         wone(1) = one
         wone(2) = one
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
      endif
c
c

      if (nint.eq.3) then
         wone(1) = five9
         wone(2) = five9
         wone(3) = eight9
         raone(1)=-.774596669241483
         raone(2)= .774596669241483
         raone(3)= zero
      endif
c
c
      if (nint.eq.4) then
         wone(1) = .347854845137454
         wone(2) = .347854845137454
         wone(3) = .652145154862546
         wone(4) = .652145154862546
         raone(1)=-.861136311594053
         raone(2)= .861136311594053
         raone(3)=-.339981043584856
         raone(4)= .339981043584856
      endif
c
c
       if(nint.eq.5) then
        wone(1) = .236926885056189
        wone(2) = .236926885056189
        wone(3) = .478628670499366
        wone(4) = .478628670499366
        wone(5) = .568888888888888
        raone(1)=-.906179845938664
        raone(2)= .906179845938664
        raone(3)=-.538469310105683
        raone(4)= .538469310105683
        raone(5)= zero
       endif
c
c
       if(nint.eq.6) then
         wone(1) = .171324492397170
         wone(2) = .171324492397170
         wone(3) = .360761573048139
         wone(4) = .360761573048139
         wone(5) = .467913934572691
         wone(6) = .467913934572691
         raone(1)=-.932469514203152
         raone(2)= .932469514203152
         raone(3)=-.661209386466265
         raone(4)= .661209386466365
         raone(5)=-.238619186083197
         raone(6)= .238619186083197
        endif
c
c
       if(nint.eq.7) then
         wone(1) = .129484966168870
         wone(2) = .129484966168870
         wone(3) = .279705391489277
         wone(4) = .279705391489277
         wone(5) = .381830050505119
         wone(6) = .381830050505119
         wone(7) = .417959183673469
         raone(1)=-.949107912342759
         raone(2)= .949107912342759
         raone(3)=-.741531185599394
         raone(4)= .741531185599394
         raone(5)=-.405845151377397
         raone(6)= .405845151377397
         raone(7)= zero
        endif
c
c
       if(nint.eq.8) then
         wone(1) = .101228536290376
         wone(2) = .101228536290376
         wone(3) = .222381034453374
         wone(4) = .222381034453374
         wone(5) = .313706645877887
         wone(6) = .313706645877887
         wone(7) = .362683783378362
         wone(8) = .362683783378362
         raone(1)=-.960289856497536
         raone(2)= .960289856497536
         raone(3)=-.796666477413627
         raone(4)= .796666477413627
         raone(5)=-.525532409916329
         raone(6)= .525532409916329
         raone(7)=-.183434642495650
         raone(8)= .183434642495650
        endif
c
c    polinomios de Legendre
c
      do 100 l = 1, nint
         r = raone(l)
c
      shlone(1,1,l) = zero
      shlone(2,1,l) = one
        if(nen.eq.1) go to 100
      shlone(1,2,l) = one
      shlone(2,2,l) = r
        if(nen.eq.2) go to 100
      shlone(1,3,l) = 3.D0 * r
      shlone(2,3,l) = 0.3D1 / 0.2D1 * r ** 2
     #              - 0.1D1 / 0.2D1
        if(nen.eq.3) go to 100
      shlone(1,4,l) = 0.15D2 / 0.2D1 * r ** 2
     #              - 0.3D1 / 0.2D1
      shlone(2,4,l) = 0.5D1 / 0.2D1 * r ** 3
     #              - 0.3D1 / 0.2D1 * r
        if(nen.eq.4) go to 100
      shlone(1,5,l) = 0.35D2 / 0.2D1 * r ** 3
     #              - 0.15D2 / 0.2D1 * r
      shlone(2,5,l) = 0.35D2 / 0.8D1 * r ** 4
     #              - 0.15D2 / 0.4D1 * r ** 2
     #              + 0.3D1 / 0.8D1
        if(nen.eq.5) go to 100
      shlone(1,6,l) = 0.315D3 / 0.8D1 * r ** 4
     #              - 0.105D3 / 0.4D1 * r ** 2
     #              + 0.15D2 / 0.8D1
      shlone(2,6,l) = 0.63D2 / 0.8D1 * r ** 5
     #              - 0.35D2 / 0.4D1 * r ** 3
     #              + 0.15D2 / 0.8D1 * r
        if(nen.eq.6) go to 100
      shlone(1,7,l) = 0.693D3 / 0.8D1 * r ** 5
     #              - 0.315D3 / 0.4D1 * r ** 3
     #              + 0.105D3 / 0.8D1 * r
      shlone(2,7,l) = 0.231D3 / 0.16D2 * r ** 6
     #              - 0.315D3 / 0.16D2 * r ** 4
     #              + 0.105D3 / 0.16D2 * r ** 2
     #              - 0.5D1 / 0.16D2
        if(nen.eq.7) go to 100
      shlone(1,8,l) = 0.3003D4 / 0.16D2 * r ** 6
     #              - 0.3465D4 / 0.16D2 * r ** 4
     #              + 0.945D3 / 0.16D2 * r ** 2
     #              - 0.35D2 / 0.16D2
      shlone(2,8,l) = 0.429D3 / 0.16D2 * r ** 7
     #              - 0.693D3 / 0.16D2 * r ** 5
     #              + 0.315D3 / 0.16D2 * r ** 3
     #              - 0.35D2 / 0.16D2 * r
c
  100  continue
c
c
      return
      end
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
      subroutine shlegdre(shl,w,nint,nen)
c
c     program to calculate integration-rule weights, shape functions
c        and local derivatives for a four-node quadrilateral element
c
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c
      implicit real*8 (a-h,o-z)
c
c     remove above card for single precision operation
c
      dimension wone(8),raone(8)
	dimension shlone(2,8,8)
      dimension shl(3,nen,*),w(*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      if (nint.eq.1) then
         wone(1)  = two
         raone(1) = zero
	nintx=1
	ninty=1
      endif
c
      if (nen.eq.1) then
	nenx=1
	neny=1
      endif
c
c
      if (nint.eq.4) then
         wone(1) = one
         wone(2) = one
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
	nintx=2
	ninty=2
      endif
c
      if (nen.eq.4) then
	nenx=2
	neny=2
      endif
c
c
c
      if (nint.eq.9) then
         wone(1) = five9
         wone(2) = five9
         wone(3) = eight9
         raone(1)=-.774596669241483
         raone(2)= .774596669241483
         raone(3)= zero
	nintx=3
	ninty=3
      endif
c
      if(nen.eq.9) then
	nenx=3
	neny=3
      endif
c
      if (nint.eq.16) then
         wone(1) = .347854845137454
         wone(2) = .347854845137454
         wone(3) = .652145154862546
         wone(4) = .652145154862546
         raone(1)=-.861136311594053
         raone(2)= .861136311594053
         raone(3)=-.339981043584856
         raone(4)= .339981043584856
	nintx=4
	ninty=4
      endif
c
      if (nen.eq.16) then
	nenx=4
	neny=4
         endif
c
c
       if(nint.eq.25) then
        wone(1) = .236926885056189
        wone(2) = .236926885056189
        wone(3) = .478628670499366
        wone(4) = .478628670499366
        wone(5) = .568888888888888
        raone(1)=-.906179845938664
        raone(2)= .906179845938664
        raone(3)=-.538469310105683
        raone(4)= .538469310105683
        raone(5)= zero
	nintx=5
	ninty=5
       endif
c
       if(nen.eq.25) then
	nenx=5
	neny=5
       endif
c
       if(nint.eq.36) then
         wone(1) = .171324492397170
         wone(2) = .171324492397170
         wone(3) = .360761573048139
         wone(4) = .360761573048139
         wone(5) = .467913934572691
         wone(6) = .467913934572691
         raone(1)=-.932469514203152
         raone(2)= .932469514203152
         raone(3)=-.661209386466265
         raone(4)= .661209386466365
         raone(5)=-.238619186083197
         raone(6)= .238619186083197
	nintx=6
	ninty=6
        endif
c
        if(nen.eq.36) then
	nenx=6
	neny=6
        endif
c
       if(nint.eq.49) then
         wone(1) = .129484966168870
         wone(2) = .129484966168870
         wone(3) = .279705391489277
         wone(4) = .279705391489277
         wone(5) = .381830050505119
         wone(6) = .381830050505119
         wone(7) = .417959183673469
         raone(1)=-.949107912342759
         raone(2)= .949107912342759
         raone(3)=-.741531185599394
         raone(4)= .741531185599394
         raone(5)=-.405845151377397
         raone(6)= .405845151377397
         raone(7)= zero
	nintx=7
	ninty=7
        endif
c
        if(nen.eq.49) then
	nenx=7
	neny=7
        endif
c
       if(nint.eq.64) then
         wone(1) = .101228536290376
         wone(2) = .101228536290376
         wone(3) = .222381034453374
         wone(4) = .222381034453374
         wone(5) = .313706645877887
         wone(6) = .313706645877887
         wone(7) = .362683783378362
         wone(8) = .362683783378362
         raone(1)=-.960289856497536
         raone(2)= .960289856497536
         raone(3)=-.796666477413627
         raone(4)= .796666477413627
         raone(5)=-.525532409916329
         raone(6)= .525532409916329
         raone(7)=-.183434642495650
         raone(8)= .183434642495650
	nintx=8
	ninty=8
        endif
        if(nen.eq.64) then
	nenx=8
	neny=8
        endif
c
c    polinomios de Legendre
c
      do 100 l = 1, nintx
         r = raone(l)
c
      shlone(1,1,l) = zero
      shlone(2,1,l) = one
        if(nenx.eq.1) go to 100
      shlone(1,2,l) = one
      shlone(2,2,l) = r
        if(nenx.eq.2) go to 100
      shlone(1,3,l) = 3.D0 * r
      shlone(2,3,l) = 0.3D1 / 0.2D1 * r ** 2 - 0.1D1 / 0.2D1
        if(nenx.eq.3) go to 100
      shlone(1,4,l) = 0.15D2 / 0.2D1 * r ** 2 - 0.3D1 / 0.2D1
      shlone(2,4,l) = 0.5D1 / 0.2D1 * r ** 3 - 0.3D1 / 0.2D1 * r
        if(nenx.eq.4) go to 100
      shlone(1,5,l) = 0.35D2 / 0.2D1 * r ** 3 - 0.15D2 / 0.2D1 * r
      shlone(2,5,l) = 0.35D2 / 0.8D1 * r ** 4 - 0.15D2 / 0.4D1 * r ** 2
     #              + 0.3D1 / 0.8D1
        if(nenx.eq.5) go to 100
      shlone(1,6,l) = 0.315D3 / 0.8D1 * r ** 4
     #              - 0.105D3 / 0.4D1 * r ** 2 + 0.15D2 / 0.8D1
      shlone(2,6,l) = 0.63D2 / 0.8D1 * r ** 5 - 0.35D2 / 0.4D1 * r ** 3
     #              + 0.15D2 / 0.8D1 * r
        if(nenx.eq.6) go to 100
      shlone(1,7,l) = 0.693D3 / 0.8D1 * r ** 5
     #              - 0.315D3 / 0.4D1 * r ** 3 + 0.105D3 / 0.8D1 * r
      shlone(2,7,l) = 0.231D3 / 0.16D2 * r ** 6
     #              - 0.315D3 / 0.16D2 * r ** 4
     #              + 0.105D3 / 0.16D2 * r ** 2 - 0.5D1 / 0.16D2
        if(nenx.eq.7) go to 100
      shlone(1,8,l) = 0.3003D4 / 0.16D2 * r ** 6
     #              - 0.3465D4 / 0.16D2 * r ** 4
     #              + 0.945D3 / 0.16D2 * r ** 2 - 0.35D2 / 0.16D2
      shlone(2,8,l) = 0.429D3 / 0.16D2 * r ** 7
     #              - 0.693D3 / 0.16D2 * r ** 5
     #              + 0.315D3 / 0.16D2 * r ** 3 - 0.35D2 / 0.16D2 * r
c
  100  continue
c
         l=0
	   do ly=1,ninty
	   do lx=1,nintx
	   l = l+1
	      w(l) = wone(lx)*wone(ly)
         end do
	   end do
c
      l=0
	do ly=1,ninty
      do lx=1,nintx
	  l = l+1
	  j=0
	 do iy=1,neny
	 do ix=1,nenx
         j = j + 1
	   shl(1,j,l) = shlone(1,ix,lx)*shlone(2,iy,ly)
	   shl(2,j,l) = shlone(2,ix,lx)*shlone(1,iy,ly)
	   shl(3,j,l) = shlone(2,ix,lx)*shlone(2,iy,ly)
	 end do
	 end do
	end do
	end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shlqpbk(shl,nen,nside,nnods,nints)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c     and local derivatives for a four-node quadrilateral element
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension raone(8),xaone(8),paone(8)
      dimension shlx(2,8),shly(2,8),inod(8,8)
      dimension shl(3,nen,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      if (nints.eq.1) then
         raone(1) = zero
         paone(1) = zero
      endif
c
      if(nnods.eq.1) then
         xaone(1) = zero
      end if
c
      if (nints.eq.2) then
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
         paone(1)= .577350269189626
         paone(2)=-.577350269189625
      endif
c
      if (nnods.eq.2) then
         xaone(1) = -one
         xaone(2) =  one
      endif
c
      if (nints.eq.3) then
         raone(1)=-.774596669241483
         raone(2)= zero
         raone(3)= .774596669241483
c
         paone(1)= .774596669241483
         paone(2)= zero
         paone(3)=-.774596669241483
      endif
c
      if(nnods.eq.3) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
      endif
c
      if (nints.eq.4) then
         raone(1)=-.861136311594053
         raone(2)=-.339981043584856
         raone(3)= .339981043584856
         raone(4)= .861136311594053
c
         paone(1)= .861136311594053
         paone(2)= .339981043584856
         paone(3)=-.339981043584856
         paone(4)=-.861136311594053
      endif
c
      if (nnods.eq.4) then
         xaone(1) = -one
         xaone(2) = one
         xaone(3) = -.333333333333333
         xaone(4) =  .333333333333333
      endif
c
      if(nints.eq.5) then
         raone(1)=-.906179845938664
         raone(2)=-.538469310105683
         raone(3)= zero
         raone(4)= .538469310105683
         raone(5)= .906179845938664
c
         paone(1)= .906179845938664
         paone(2)= .538469310105683
         paone(3)= zero
         paone(4)=-.538469310105683
         paone(5)=-.906179845938664
c
      endif
c
      if(nnods.eq.5) then
         xaone(1)= -one
         xaone(2)=  one
         xaone(3)= -pt5
         xaone(4)= zero
         xaone(5)= pt5
      endif
c
      if(nints.eq.6) then
         raone(1)=-.932469514203152
         raone(2)=-.661209386466265
         raone(3)=-.238619186083197
         raone(4)= .238619186083197
         raone(5)= .661209386466365
         raone(6)= .932469514203152
c
         paone(1)= .932469514203152
         paone(2)= .661209386466265
         paone(3)= .238619186083197
         paone(4)=-.238619186083197
         paone(5)=-.661209386466365
         paone(6)=-.932469514203152
      endif
c
      if(nnods.eq.6) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.600000000000000
         xaone(4) = -.200000000000000
         xaone(5) =  .200000000000000
         xaone(6) =  .600000000000000
      endif
c
      if(nints.eq.7) then
         raone(1)=-.949107912342759
         raone(2)=-.741531185599394
         raone(3)=-.405845151377397
         raone(4)= zero
         raone(5)= .405845151377397
         raone(6)= .741531185599394
         raone(7)= .949107912342759
c
         paone(1)= .949107912342759
         paone(2)= .741531185599394
         paone(3)= .405845151377397
         paone(4)= zero
         paone(5)=-.405845151377397
         paone(6)=-0.741531185599394
         paone(7)=-.949107912342759
c
      endif
c
      if(nnods.eq.7) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.666666666666666
         xaone(4) = -.333333333333333
         xaone(5) = zero
         xaone(6) =  .333333333333333
         xaone(7) =  .666666666666666
      endif
c
      if(nints.eq.8) then
         raone(1)=-.960289856497536
         raone(2)=-.796666477413627
         raone(3)=-.525532409916329
         raone(4)=-.183434642495650
         raone(5)= .183434642495650
         raone(6)= .525532409916329
         raone(7)= .796666477413627
         raone(8)= .960289856497536
c
         paone(1)= .960289856497536
         paone(2)= .796666477413627
         paone(3)= .525532409916329
         paone(4)= .183434642495650
         paone(5)=-.183434642495650
         paone(6)=-.525532409916329
         paone(7)=-.796666477413627
         paone(8)=-.960289856497536
c
      endif
      if(nnods.eq.8) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -0.71428571428571
         xaone(4) = -0.42857142857143
         xaone(5) = -0.14285714285714
         xaone(6) = 0.14285714285714
         xaone(7) = 0.42857142857143
         xaone(8) = 0.71428571428571
      endif
c
c     trata nen
c
      if(nen.eq.1) inod(1,1) = 1
c
      if(nen.eq.4) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(1,2) = 4
         inod(2,2) = 3
      end if
c
      if(nen.eq.9) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 7
c
         inod(1,3) = 8
         inod(2,3) = 6
         inod(3,3) = 9
      end if
c
      if(nen.eq.16) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 10
         inod(4,2) = 9
c
         inod(1,3) = 12
         inod(2,3) = 7
         inod(3,3) = 13
         inod(4,3) = 14
c
         inod(1,4) = 11
         inod(2,4) = 8
         inod(3,4) = 16
         inod(4,4) = 15
      end if
c
c
      if(nen.eq.25) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 13
         inod(4,2) = 12
         inod(5,2) = 11
c
         inod(1,3) = 16
         inod(2,3) = 8
         inod(3,3) = 17
         inod(4,3) = 18
         inod(5,3) = 19
c
         inod(1,4) = 15
         inod(2,4) = 9
         inod(3,4) = 24
         inod(4,4) = 25
         inod(5,4) = 20
c
         inod(1,5) = 14
         inod(2,5) = 10
         inod(3,5) = 23
         inod(4,5) = 22
         inod(5,5) = 21
      end if
c
      if(nen.eq.36) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 16
         inod(4,2) = 15
         inod(5,2) = 14
         inod(6,2) = 13
c
         inod(1,3) = 20
         inod(2,3) = 9
         inod(3,3) = 21
         inod(4,3) = 22
         inod(5,3) = 23
         inod(6,3) = 24
c
         inod(1,4) = 19
         inod(2,4) = 10
         inod(3,4) = 32
         inod(4,4) = 33
         inod(5,4) = 34
         inod(6,4) = 25
c
         inod(1,5) = 18
         inod(2,5) = 11
         inod(3,5) = 31
         inod(4,5) = 36
         inod(5,5) = 35
         inod(6,5) = 26
c
         inod(1,6) = 17
         inod(2,6) = 12
         inod(3,6) = 30
         inod(4,6) = 29
         inod(5,6) = 28
         inod(6,6) = 27
c
      end if
c
      if(nen.eq.49) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
         inod(7,1) = 9
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 19
         inod(4,2) = 18
         inod(5,2) = 17
         inod(6,2) = 16
         inod(7,2) = 15
c
         inod(1,3) = 24
         inod(2,3) = 10
         inod(3,3) = 25
         inod(4,3) = 26
         inod(5,3) = 27
         inod(6,3) = 28
         inod(7,3) = 29
c
         inod(1,4) = 23
         inod(2,4) = 11
         inod(3,4) = 40
         inod(4,4) = 41
         inod(5,4) = 42
         inod(6,4) = 43
         inod(7,4) = 30
c
         inod(1,5) = 22
         inod(2,5) = 12
         inod(3,5) = 39
         inod(4,5) = 48
         inod(5,5) = 49
         inod(6,5) = 44
         inod(7,5) = 31
c
         inod(1,6) = 21
         inod(2,6) = 13
         inod(3,6) = 38
         inod(4,6) = 47
         inod(5,6) = 46
         inod(6,6) = 45
         inod(7,6) = 32
c
         inod(1,7) = 20
         inod(2,7) = 14
         inod(3,7) = 37
         inod(4,7) = 36
         inod(5,7) = 35
         inod(6,7) = 34
         inod(7,7) = 33
c
      end if
c
c
      if(nen.eq.64) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
         inod(7,1) = 9
         inod(8,1) = 10
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 22
         inod(4,2) = 21
         inod(5,2) = 20
         inod(6,2) = 19
         inod(7,2) = 18
         inod(8,2) = 17
c
         inod(1,3) = 28
         inod(2,3) = 11
         inod(3,3) = 29
         inod(4,3) = 30
         inod(5,3) = 31
         inod(6,3) = 32
         inod(7,3) = 33
         inod(8,3) = 34
c
         inod(1,4) = 27
         inod(2,4) = 12
         inod(3,4) = 48
         inod(4,4) = 49
         inod(5,4) = 50
         inod(6,4) = 51
         inod(7,4) = 52
         inod(8,4) = 35
c
         inod(1,5) = 26
         inod(2,5) = 13
         inod(3,5) = 47
         inod(4,5) = 60
         inod(5,5) = 61
         inod(6,5) = 62
         inod(7,5) = 53
         inod(8,5) = 36
c
         inod(1,6) = 25
         inod(2,6) = 14
         inod(3,6) = 46
         inod(4,6) = 59
         inod(5,6) = 64
         inod(6,6) = 63
         inod(7,6) = 54
         inod(8,6) = 37
c
         inod(1,7) = 24
         inod(2,7) = 15
         inod(3,7) = 45
         inod(4,7) = 58
         inod(5,7) = 57
         inod(6,7) = 56
         inod(7,7) = 55
         inod(8,7) = 38
c
         inod(1,8) = 23
         inod(2,8) = 16
         inod(3,8) = 44
         inod(4,8) = 43
         inod(5,8) = 42
         inod(6,8) = 41
         inod(7,8) = 40
         inod(8,8) = 39
c
      end if
c
      lb=0
      do 200 ns=1,nside
         do 200 l = 1, nints
            lb = lb + 1
c     aresta 1
            if(ns.eq.1) then
               r = raone(l)
               s = -one
            end if
c     aresta 2
            if(ns.eq.2) then
               r = one
               s = raone(l)
            end if
c     aresta 3
            if(ns.eq.3) then
               r = paone(l)
               s = one
            end if
c     aresta 4
            if(ns.eq.4) then
               r = -one
               s = paone(l)
            end if
c
            shlx(1,1) = zero
            shly(1,1) = zero
            shlx(2,1) = one
            shly(2,1) = one
c
            if(nnods.eq.1) go to 100
c
            do i = 1, nnods
               aa = one
               bb = one
               cc = one
               aax= zero
               aay= zero
               do j =1, nnods
                  daj = one
                  caj = one
                  if (i .ne. j)then
                     aa = aa * ( r - xaone(j))
                     cc = cc * ( s - xaone(j))
                     bb = bb * ( xaone(i) - xaone(j))
                     do k = 1, nnods
                        if(k.ne.i.and.k.ne.j) daj=daj*(r-xaone(k))
                        if(k.ne.i.and.k.ne.j) caj=caj*(s-xaone(k))
                     end do
                     aax = aax + daj
                     aay = aay + caj
                  endif
               end do
               shlx(1,i) = aax/bb
               shly(1,i) = aay/bb
               shlx(2,i) = aa/bb
               shly(2,i) = cc/bb
            end do
c
 100        continue
c
            do iy=1,nnods
               do ix=1,nnods
                  j = inod(ix,iy)
                  shl(1,j,lb) = shlx(1,ix)*shly(2,iy)
                  shl(2,j,lb) = shlx(2,ix)*shly(1,iy)
                  shl(3,j,lb) = shlx(2,ix)*shly(2,iy)
               end do
            end do
 200     continue
c
         return
         end

c-----------------------------------------------------------------------
      subroutine shlqpk(shl,w,nint,nen)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c     and local derivatives for a four-node QUADrilateral element
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension wone(8),raone(8),xaone(8)
      dimension shlone(2,8,8),inod(8,8)
      dimension shl(3,nen,*),w(*)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      if (nint.eq.1) then
         wone(1)  = two
         raone(1) = zero
         nintx=1
         ninty=1
      endif
c
      if (nen.eq.1) then
         xaone(1) = zero
	 nenx=1
	 neny=1
      end if
c
      if (nint.eq.4) then
         wone(1) = one
         wone(2) = one
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
         nintx=2
         ninty=2
      endif
c
      if (nen.eq.4) then
         xaone(1) = -one
         xaone(2) =  one
         nenx=2
         neny=2
      endif
c
      if (nint.eq.9) then
c$$$         wone(1) = five9
c$$$         wone(2) = five9
c$$$         wone(3) = eight9
c$$$         raone(1)=-.774596669241483
c$$$         raone(2)= .774596669241483
c$$$         raone(3)= zero
c$$$         nintx=3
c$$$  ninty=3
         wone(1) = five9
         wone(2) = eight9
         wone(3) = five9
         raone(1)=-.774596669241483
         raone(2)= zero
         raone(3)= .774596669241483
         nintx=3
         ninty=3
      endif
c
      if(nen.eq.9) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
         nenx=3
         neny=3
      endif
c
      if (nint.eq.16) then
         wone(1) = .347854845137454
         wone(2) = .347854845137454
         wone(3) = .652145154862546
         wone(4) = .652145154862546
         raone(1)=-.861136311594053
         raone(2)= .861136311594053
         raone(3)=-.339981043584856
         raone(4)= .339981043584856
         nintx=4
         ninty=4
      endif
c
      if (nen.eq.16) then
         xaone(1) = -one
         xaone(2) = one
         xaone(3) = -.333333333333333
         xaone(4) =  .333333333333333
         nenx=4
         neny=4
      endif
c
      if(nint.eq.25) then
         wone(1) = .236926885056189
         wone(2) = .236926885056189
         wone(3) = .478628670499366
         wone(4) = .478628670499366
         wone(5) = .568888888888888
         raone(1)=-.906179845938664
         raone(2)= .906179845938664
         raone(3)=-.538469310105683
         raone(4)= .538469310105683
         raone(5)= zero
         nintx=5
         ninty=5
      endif
c
      if(nen.eq.25) then
         xaone(1)= -one
         xaone(2)=  one
         xaone(3)= -pt5
         xaone(4)= zero
         xaone(5)= pt5
         nenx=5
         neny=5
      endif
c
      if(nint.eq.36) then
         wone(1) = .171324492397170
         wone(2) = .171324492397170
         wone(3) = .360761573048139
         wone(4) = .360761573048139
         wone(5) = .467913934572691
         wone(6) = .467913934572691
         raone(1)=-.932469514203152
         raone(2)= .932469514203152
         raone(3)=-.661209386466265
         raone(4)= .661209386466365
         raone(5)=-.238619186083197
         raone(6)= .238619186083197
         nintx=6
         ninty=6
      endif
c
      if(nen.eq.36) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.600000000000000
         xaone(4) = -.200000000000000
         xaone(5) =  .200000000000000
         xaone(6) =  .600000000000000
         nenx=6
         neny=6
      endif
c
      if(nint.eq.49) then
         wone(1) = .129484966168870
         wone(2) = .129484966168870
         wone(3) = .279705391489277
         wone(4) = .279705391489277
         wone(5) = .381830050505119
         wone(6) = .381830050505119
         wone(7) = .417959183673469
         raone(1)=-.949107912342759
         raone(2)= .949107912342759
         raone(3)=-.741531185599394
         raone(4)= .741531185599394
         raone(5)=-.405845151377397
         raone(6)= .405845151377397
         raone(7)= zero
         nintx=7
         ninty=7
      endif
c
      if(nen.eq.49) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.666666666666666
         xaone(4) = -.333333333333333
         xaone(5) = zero
         xaone(6) =  .333333333333333
         xaone(7) =  .666666666666666
         nenx=7
         neny=7
      endif
c
      if(nint.eq.64) then
         wone(1) = .101228536290376
         wone(2) = .101228536290376
         wone(3) = .222381034453374
         wone(4) = .222381034453374
         wone(5) = .313706645877887
         wone(6) = .313706645877887
         wone(7) = .362683783378362
         wone(8) = .362683783378362
         raone(1)=-.960289856497536
         raone(2)= .960289856497536
         raone(3)=-.796666477413627
         raone(4)= .796666477413627
         raone(5)=-.525532409916329
         raone(6)= .525532409916329
         raone(7)=-.183434642495650
         raone(8)= .183434642495650
         nintx=8
         ninty=8
      endif
c
      if(nen.eq.64) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -0.71428571428571
         xaone(4) = -0.42857142857143
         xaone(5) = -0.14285714285714
         xaone(6) = 0.14285714285714
         xaone(7) = 0.42857142857143
         xaone(8) = 0.71428571428571
         nenx=8
         neny=8
      endif
c
      if(nen.eq.1) inod(1,1) = 1
c
      if(nen.eq.4) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(1,2) = 4
         inod(2,2) = 3
      end if
c
      if(nen.eq.9) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 7
c
         inod(1,3) = 8
         inod(2,3) = 6
         inod(3,3) = 9
      end if
c
      if(nen.eq.16) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 10
         inod(4,2) = 9
c
         inod(1,3) = 12
         inod(2,3) = 7
         inod(3,3) = 13
         inod(4,3) = 14
c
         inod(1,4) = 11
         inod(2,4) = 8
         inod(3,4) = 16
         inod(4,4) = 15
      end if
c
      if(nen.eq.25) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 13
         inod(4,2) = 12
         inod(5,2) = 11
c
         inod(1,3) = 16
         inod(2,3) = 8
         inod(3,3) = 17
         inod(4,3) = 18
         inod(5,3) = 19
c
         inod(1,4) = 15
         inod(2,4) = 9
         inod(3,4) = 24
         inod(4,4) = 25
         inod(5,4) = 20
c
         inod(1,5) = 14
         inod(2,5) = 10
         inod(3,5) = 23
         inod(4,5) = 22
         inod(5,5) = 21
      end if
c
c
      if(nen.eq.36) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 16
         inod(4,2) = 15
         inod(5,2) = 14
         inod(6,2) = 13
c
         inod(1,3) = 20
         inod(2,3) = 9
         inod(3,3) = 21
         inod(4,3) = 22
         inod(5,3) = 23
         inod(6,3) = 24
c
         inod(1,4) = 19
         inod(2,4) = 10
         inod(3,4) = 32
         inod(4,4) = 33
         inod(5,4) = 34
         inod(6,4) = 25
c
         inod(1,5) = 18
         inod(2,5) = 11
         inod(3,5) = 31
         inod(4,5) = 36
         inod(5,5) = 35
         inod(6,5) = 26
c
         inod(1,6) = 17
         inod(2,6) = 12
         inod(3,6) = 30
         inod(4,6) = 29
         inod(5,6) = 28
         inod(6,6) = 27
      end if
c
      if(nen.eq.49) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
         inod(7,1) = 9
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 19
         inod(4,2) = 18
         inod(5,2) = 17
         inod(6,2) = 16
         inod(7,2) = 15
c
         inod(1,3) = 24
         inod(2,3) = 10
         inod(3,3) = 25
         inod(4,3) = 26
         inod(5,3) = 27
         inod(6,3) = 28
         inod(7,3) = 29
c
         inod(1,4) = 23
         inod(2,4) = 11
         inod(3,4) = 40
         inod(4,4) = 41
         inod(5,4) = 42
         inod(6,4) = 43
         inod(7,4) = 30
c
         inod(1,5) = 22
         inod(2,5) = 12
         inod(3,5) = 39
         inod(4,5) = 48
         inod(5,5) = 49
         inod(6,5) = 44
         inod(7,5) = 31
c
         inod(1,6) = 21
         inod(2,6) = 13
         inod(3,6) = 38
         inod(4,6) = 47
         inod(5,6) = 46
         inod(6,6) = 45
         inod(7,6) = 32
c
         inod(1,7) = 20
         inod(2,7) = 14
         inod(3,7) = 37
         inod(4,7) = 36
         inod(5,7) = 35
         inod(6,7) = 34
         inod(7,7) = 33
      end if
c
      if(nen.eq.64) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
         inod(7,1) = 9
         inod(8,1) = 10
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 22
         inod(4,2) = 21
         inod(5,2) = 20
         inod(6,2) = 19
         inod(7,2) = 18
         inod(8,2) = 17
c
         inod(1,3) = 28
         inod(2,3) = 11
         inod(3,3) = 29
         inod(4,3) = 30
         inod(5,3) = 31
         inod(6,3) = 32
         inod(7,3) = 33
         inod(8,3) = 34
c
         inod(1,4) = 27
         inod(2,4) = 12
         inod(3,4) = 48
         inod(4,4) = 49
         inod(5,4) = 50
         inod(6,4) = 51
         inod(7,4) = 52
         inod(8,4) = 35
c
         inod(1,5) = 26
         inod(2,5) = 13
         inod(3,5) = 47
         inod(4,5) = 60
         inod(5,5) = 61
         inod(6,5) = 62
         inod(7,5) = 53
         inod(8,5) = 36
c
         inod(1,6) = 25
         inod(2,6) = 14
         inod(3,6) = 46
         inod(4,6) = 59
         inod(5,6) = 64
         inod(6,6) = 63
         inod(7,6) = 54
         inod(8,6) = 37
c
         inod(1,7) = 24
         inod(2,7) = 15
         inod(3,7) = 45
         inod(4,7) = 58
         inod(5,7) = 57
         inod(6,7) = 56
         inod(7,7) = 55
         inod(8,7) = 38
c
         inod(1,8) = 23
         inod(2,8) = 16
         inod(3,8) = 44
         inod(4,8) = 43
         inod(5,8) = 42
         inod(6,8) = 41
         inod(7,8) = 40
         inod(8,8) = 39
c
      end if
c
c     loop integration points to evaluate functions/deriv
c
      do 100 l = 1, nintx
         r = raone(l)
c
         write(*,'(A,I5,F8.4)') " ponto de gauss (shlone)", l,r
c
         if(nenx.eq.1) then
            shlone(1,1,l) = zero
            shlone(2,1,l) = one
            go to 100
         endif
c
         do i = 1, nenx
            aa = one
            bb = one
            aax = zero
            do j =1, nenx
               daj = one
               if (i .ne. j)then
                  aa = aa * ( r - xaone(j))
                  bb = bb * ( xaone(i) - xaone(j))
                  do k = 1, nenx
                     if(k.ne.i.and.k.ne.j) daj = daj * (r-xaone(k))
                  end do
                  aax = aax + daj
               endif
            end do
            shlone(2,i,l) = aa/bb
            shlone(1,i,l) = aax/bb
         end do
c
 100  continue
c
      l=0
      do ly=1,ninty
         do lx=1,nintx
            l = l+1
            w(l) = wone(lx)*wone(ly)
         end do
      end do
c
      l=0
      do ly=1,ninty
         do lx=1,nintx
            l = l+1
            r = raone(lx)
            s = raone(ly)
            write(*,'(A,3I5,2F8.4)') " ponto de gauss", l,lx,ly,r,s
            do iy=1,neny
               do ix=1,nenx
                  j = inod(ix,iy)
                  shl(1,j,l) = shlone(1,ix,lx)*shlone(2,iy,ly)
                  shl(2,j,l) = shlone(2,ix,lx)*shlone(1,iy,ly)
                  shl(3,j,l) = shlone(2,ix,lx)*shlone(2,iy,ly)
               end do
            end do
c
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shlhxpk(shl,w,nint,nen)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c     and local derivatives for a HEXAhedral element
c             r,s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local ("zeta") derivative of shape function
c        shl(4,i,l) = local shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension wone(8),raone(8),xaone(8)
      dimension shlone(2,8,8),inod(8,8,8)
      dimension shl(4,nen,*),w(*),ra(64),sa(64),ta(64)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      write(*,'(A,I5)') " nint=", nint
c
      if (nint.eq.1) then
         wone(1)  = two
         raone(1) = zero
         nintx = 1
         ninty = 1
         nintz = 1
      endif
c
      if (nen.eq.1) xaone(1) = zero
c
      if (nint.eq.8) then
         wone(1) = one
         wone(2) = one
         raone(1) =-.577350269189626
         raone(2) = .577350269189625
         nintx = 2
         ninty = 2
         nintz = 2
      endif
c
      if (nint.eq.27) then
         wone(1) = five9
         wone(2) = eight9
         wone(3) = five9
         raone(1) = -.774596669241483
         raone(2) = zero
         raone(3) = .774596669241483
         nintx = 3
         ninty = 3
         nintz = 3
      end if
c
      if (nen.eq.8) then
         xaone(1) = -one
         xaone(2) =  one
         nenx=2
         neny=2
         nenz=2
      endif
c
c     inod - specify the ordering of the nodes/shape functions
c     it follows xaone
c
      if(nen.eq.1) inod(1,1,1) = 1
c
      if(nen.eq.8) then
         inod(1,1,1) = 1
         inod(2,1,1) = 2
         inod(1,2,1) = 4
         inod(2,2,1) = 3
c
         inod(1,1,2) = 5
         inod(2,1,2) = 6
         inod(1,2,2) = 8
         inod(2,2,2) = 7
      end if
c
c     generate 1d shape functions
c
      do 100 l = 1, nintx
         r = raone(l)
c
         if(nenx.eq.1) then
            shlone(1,1,l) = zero
            shlone(2,1,l) = one
            go to 100
         endif
c
         do i = 1, nenx
            aa = one
            bb = one
            aax = zero
            do j = 1, nenx
               daj = one
               if (i .ne. j) then
                  aa = aa * ( r - xaone(j))
                  bb = bb * ( xaone(i) - xaone(j))
                  do k = 1, nenx
                     if(k.ne.i.and.k.ne.j) daj=daj*(r-xaone(k))
                  end do
                  aax = aax + daj
               endif
            end do
            shlone(2,i,l) = aa/bb
            shlone(1,i,l) = aax/bb
         end do
 100  continue

      write(*,*) "debug shlone"
      do i=1,nenx
         write(*,'(10F8.4)') (shlone(2,i,l),l=1,nintx)
      end do

c
c     generates weights of the gauss quadrature / and local r,s,t nodes
c
      l=0
      do lz=1,nintz
         do ly=1,ninty
            do lx=1,nintx
               l = l+1
               w(l)  = wone(lx)*wone(ly)*wone(lz)
               ra(l) = raone(lx)
               sa(l) = raone(ly)
               ta(l) = raone(lz)
            end do
         end do
      end do

c      write(*,*) "debug gauss points"
c      nl = nintx*ninty*nintz
c      do l=1,nl
c         write(*,*) ra(l), sa(l), ta(l)
c      end do

c
c     generates 3d shape functions by tensor product
c
      l=0
      do lz=1,nintz
         do ly=1,ninty
            do lx=1,nintx
c
               l=l+1
c
               write(*,'(A,4I5,3F8.4)') " ponto de gauss",
     &                              l,lx,ly,lz,ra(l),sa(l),ta(l)
c
               do iz=1,nenz
                  do iy=1,neny
                     do ix=1,nenx
c
                        j = inod(ix,iy,iz)
                        shl(1,j,l) = shlone(1,ix,lx)*
     &                               shlone(2,iy,ly)*
     &                               shlone(2,iz,lz)
c
                        shl(2,j,l) = shlone(2,ix,lx)*
     &                               shlone(1,iy,ly)*
     &                               shlone(2,iz,lz)
c
                        shl(3,j,l) = shlone(2,ix,lx)*
     &                               shlone(2,iy,ly)*
     &                               shlone(1,iz,lz)
c
c                  xaux = shlone(2,ix,lx)*shlone(2,iy,ly)*shlone(2,iz,lz)
c                  write(*,*) l, j, xaux

                        shl(4,j,l) = shlone(2,ix,lx)*
     &                               shlone(2,iy,ly)*
     &                               shlone(2,iz,lz)
                     end do
                  end do
               end do
c
            end do
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shlhxpbk(shl,nen,nside,nnods,nints)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c     and local derivatives for the faces of a eight-node
c     hexahedral element
c-----------------------------------------------------------------------
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local ("zeta") derivative of shape function
c        shl(4,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension raone(8),xaone(8),taone(8),paone(8)
      dimension shlx(2,8),shly(2,8),shlz(2,8),inod(8,8,8)
      dimension shl(4,nen,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      dimension ra(4),sa(4)
c
      write(*,'(A,I5)') " nen  :", nen
      write(*,'(A,I5)') " nnods:", nnods
      write(*,'(A,I5)') " nints:", nints
c
c     em uso
c     nints=4, nnods=2, nen=8
c

c
      if (nints.eq.1) then
         nintx = 1
         ninty = 1
         raone(1) = zero
         paone(1) = zero
         taone(1) = zero
      endif
c
      if(nnods.eq.1) then
         xaone(1) = zero
      end if
c
      if (nints.eq.2) then
         nintx=2
         ninty=2
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
c
         paone(1)=-.577350269189626
         paone(2)= .577350269189625
c
         taone(1)=-.577350269189626
         taone(2)= .577350269189625
      endif
c
      if (nnods.eq.2) then
         xaone(1) = -one
         xaone(2) =  one
      endif
c
      if (nints.eq.3) then
         nintx=3
         ninty=3

         raone(1)=-.774596669241483
         raone(2)= zero
         raone(3)= .774596669241483
c
         paone(1)= -.774596669241483
         paone(2)= zero
         paone(3)= .774596669241483
c
         taone(1)= -.774596669241483
         taone(2)= zero
         taone(3)= .774596669241483
      endif
c
      if(nnods.eq.3) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
      endif
c
      if(nints.gt.3) then
         write(*,*) "error nints shlhxpbk"
         stop
      end if
      if(nnods.gt.2) then
         write(*,*) "error nnods shlhxpbk"
         stop
      end if
c
c     trata nen
c
      if(nen.eq.1) inod(1,1,1) = 1
c
      if(nen.eq.8) then
         inod(1,1,1) = 1
         inod(2,1,1) = 2
         inod(1,2,1) = 4
         inod(2,2,1) = 3
c
         inod(1,1,2) = 5
         inod(2,1,2) = 6
         inod(1,2,2) = 8
         inod(2,2,2) = 7
      end if
c
c     build shape functions/derivatives
c
      write(*,*)
      write(*,*) "building shl on faces"
c
c     not in use
c
      l = 0
      do ly = 1,ninty
         do lx = 1,nintx
            l = l + 1
            ra(l) = raone(lx)
            sa(l) = raone(ly)
         end do
      end do
c
c     loop on integration points
c
      lb=0
      do ns=1,nside
         write(*,'(A,I5)') " face",ns
         do ly = 1, ninty
            do lx = 1, nintx

               lb = lb + 1

c     frente
               if(ns.eq.1) then
                  r = raone(lx)
                  s = -one
                  t = taone(ly)
               end if
c     esq
               if(ns.eq.2) then
                  r = -one
                  s = paone(lx)
                  t = taone(ly)
               end if
c     dir
               if(ns.eq.3) then
                  r = one
                  s = paone(lx)
                  t = taone(ly)
               end if
c     tras
               if(ns.eq.4) then
                  r = raone(lx)
                  s = one
                  t = taone(ly)
               end if
c     baixo
               if(ns.eq.5) then
                  r = raone(lx)
                  s = paone(ly)
                  t = -one
               end if
c     cima
               if(ns.eq.6) then
                  r = raone(lx)
                  s = paone(ly)
                  t = one
               end if
c
               write(*,'(A,3I5,3F8.4)') " ponto gauss", lx,ly,lb,r,s,t
c
               shlx(1,1) = zero
               shly(1,1) = zero
               shlz(1,1) = zero
c
               shlx(2,1) = one
               shly(2,1) = one
               shlz(2,1) = one
c
               if(nnods.eq.1) go to 100
c
               do i = 1, nnods
                  aa = one
                  bb = one
                  cc = one
                  dd = one
                  aax = zero
                  aay = zero
                  aaz = zero
                  do j =1, nnods
                     daj = one
                     caj = one
                     baj = one
                     if (i.ne.j) then
                        aa = aa * ( r - xaone(j))
                        cc = cc * ( s - xaone(j))
                        dd = dd * ( t - xaone(j))
                        bb = bb * ( xaone(i) - xaone(j))
                        do k = 1, nnods
                           if(k.ne.i.and.k.ne.j) daj = daj*(r-xaone(k))
                           if(k.ne.i.and.k.ne.j) caj = caj*(s-xaone(k))
                           if(k.ne.i.and.k.ne.j) baj = baj*(t-xaone(k))
                        end do
                        aax = aax + daj
                        aay = aay + caj
                        aaz = aaz + baj
                     endif
                  end do
                  shlx(1,i) = aax/bb
                  shly(1,i) = aay/bb
                  shlz(1,i) = aaz/bb
c
                  shlx(2,i) = aa/bb
                  shly(2,i) = cc/bb
                  shlz(2,i) = dd/bb
               end do
c
 100           continue
c
c               write(*,'(A,3F8.4)') " ponto de gauss", r,s,t
c
               do iz=1,nnods
                  do iy=1,nnods
                     do ix=1,nnods
                        j = inod(ix,iy,iz)
                        shl(1,j,lb) = shlx(1,ix)*shly(2,iy)*shlz(2,iz)
                        shl(2,j,lb) = shlx(2,ix)*shly(1,iy)*shlz(2,iz)
                        shl(3,j,lb) = shlx(2,ix)*shly(2,iy)*shlz(1,iz)
c
                        shl(4,j,lb) = shlx(2,ix)*shly(2,iy)*shlz(2,iz)
                     end do
                  end do
               end do
c
            end do
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shlhxpbkOold(shl,nen,nside,nnods,nints)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c     and local derivatives for the faces of a eight-node
c     hexahedral element
c-----------------------------------------------------------------------
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local ("zeta") derivative of shape function
c        shl(4,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension raone(8),xaone(8),taone(8),paone(8)
      dimension shlx(2,8),shly(2,8),shlz(2,8),inod(8,8,8)
      dimension shl(4,nen,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      write(*,*) "debug shlhxpbk"
      write(*,'(A,I5)') "nen  :", nen
      write(*,'(A,I5)') "nnods:", nnods
      write(*,'(A,I5)') "nints:", nints
c
c     em uso
c     nints=4
c     nnods=2
c     nen=8
c

c
      if (nints.eq.1) then
         raone(1) = zero
         paone(1) = zero
      endif
c
      if(nnods.eq.1) then
         xaone(1) = zero
      end if
c
      if (nints.eq.2) then
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
c
         paone(1)= .577350269189626
         paone(2)=-.577350269189625
c
         taone(1)= .577350269189626
         taone(2)=-.577350269189625
      endif
c
      if (nnods.eq.2) then
         xaone(1) = -one
         xaone(2) =  one
      endif
c
      if (nints.eq.3) then
         raone(1)=-.774596669241483
         raone(2)= zero
         raone(3)= .774596669241483
c
         paone(1)= .774596669241483
         paone(2)= zero
         paone(3)=-.774596669241483
c
         taone(1)= .774596669241483
         taone(2)= zero
         taone(3)=-.774596669241483
      endif
c
      if(nnods.eq.3) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
      endif
c
      if (nints.eq.4) then
         raone(1)=-.861136311594053
         raone(2)=-.339981043584856
         raone(3)= .339981043584856
         raone(4)= .861136311594053
c
         paone(1)= .861136311594053
         paone(2)= .339981043584856
         paone(3)=-.339981043584856
         paone(4)=-.861136311594053
c
         taone(1)= .861136311594053
         taone(2)= .339981043584856
         taone(3)=-.339981043584856
         taone(4)=-.861136311594053
      endif
c
      if (nnods.eq.4) then
         xaone(1) = -one
         xaone(2) = one
         xaone(3) = -.333333333333333
         xaone(4) =  .333333333333333
      endif
c
c
c
      if(nen.eq.1) inod(1,1,1) = 1
c
      if(nen.eq.8) then
         inod(1,1,1) = 1
         inod(2,1,1) = 2
         inod(1,2,1) = 4
         inod(2,2,1) = 3
c
         inod(1,1,2) = 5
         inod(2,1,2) = 6
         inod(1,2,2) = 8
         inod(2,2,2) = 7
      end if
c
c     build shape functions/derivatives
c
      lb=0
      do ns=1,nside
         do l = 1, nints
            lb = lb + 1
c     frente
            if(ns.eq.1) then
               r = raone(l)
               s = -one
               t = taone(l)
            end if
c     esq
            if(ns.eq.2) then
               r = -one
               s = paone(l)
               t = taone(l)
            end if
c     dir
            if(ns.eq.3) then
               r = one
               s = raone(l)
               t = taone(l)
            end if
c     tras
            if(ns.eq.4) then
               r = raone(l)
               s = one
               t = taone(l)
            end if
c     baixo
            if(ns.eq.5) then
               r = raone(l)
               s = paone(l)
               t = -one
            end if
c     cima
            if(ns.eq.6) then
               r = raone(l)
               s = paone(l)
               t = one
            end if
c
            shlx(1,1) = zero
            shly(1,1) = zero
            shlz(1,1) = zero
            shlx(2,1) = one
            shly(2,1) = one
            shlz(2,1) = one
c
            if(nnods.eq.1) go to 100
c
            do i = 1, nnods
               aa = one
               bb = one
               cc = one
               dd = one
               aax = zero
               aay = zero
               aaz = zero
               do j =1, nnods
                  daj = one
                  caj = one
                  baj = one
                  if (i.ne.j) then
                     aa = aa * ( r - xaone(j))
                     cc = cc * ( s - xaone(j))
                     dd = dd * ( t - xaone(j))
                     bb = bb * ( xaone(i) - xaone(j))
                     do k = 1, nnods
                        if(k.ne.i.and.k.ne.j) daj = daj*(r-xaone(k))
                        if(k.ne.i.and.k.ne.j) caj = caj*(s-xaone(k))
                        if(k.ne.i.and.k.ne.j) baj = baj*(t-xaone(k))
                     end do
                     aax = aax + daj
                     aay = aay + caj
                     aaz = aaz + baj
                  endif
               end do
               shlx(1,i) = aax/bb
               shly(1,i) = aay/bb
               shlz(1,i) = aaz/bb
c
               shlx(2,i) = aa/bb
               shly(2,i) = cc/bb
               shlz(2,i) = dd/bb
            end do
c
 100        continue
c
            do iz=1,nnods
               do iy=1,nnods
                  do ix=1,nnods
                     j = inod(ix,iy,iz)
                     shl(1,j,lb) = shlx(1,ix)*shly(2,iy)*shlz(2,iz)
                     shl(2,j,lb) = shlx(2,ix)*shly(1,iy)*shlz(2,iz)
                     shl(3,j,lb) = shlx(2,ix)*shly(2,iy)*shlz(1,iz)
                     shl(4,j,lb) = shlx(2,ix)*shly(2,iy)*shlz(2,iz)
                  end do
               end do
            end do
c
         end do
      end do
c
      return
      end

c-----------------------------------------------------------------------
      subroutine shlhxpbk2(shl,nen,nside,nnods,nints)
c-----------------------------------------------------------------------
c     program to calculate integration-rule weights, shape functions
c     and local derivatives for the faces of a eight-node
c     hexahedral element
c-----------------------------------------------------------------------
c               s,t = local element coordinates ("xi", "eta", resp.)
c        shl(1,i,l) = local ("xi") derivative of shape function
c        shl(2,i,l) = local ("eta") derivative of shape function
c        shl(3,i,l) = local ("zeta") derivative of shape function
c        shl(4,i,l) = local  shape function
c              w(l) = integration-rule weight
c                 i = local node number
c                 l = integration point number
c              nint = number of integration points, eq. 1 or 4
c-----------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension raone(8),paone(8),taone(8),xaone(8)
      dimension shlx(2,8),shly(2,8),shlz(2,8),inod(8,8,8)
      dimension shl(4,nen,*)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      data  five9/0.5555555555555555d0/,eight9/0.8888888888888888d0/
      data r1/0.d00/,w1/2.d00/,
     &     r2/0.577350269189626d00/,w2/1.d00/,
     &     r3a/0.774596669241483d00/,w3a/0.555555555555556d00/,
     &     r3b/0.d00/,w3b/0.888888888888889d00/,
     &     r4a/0.861136311594053d00/,w4a/0.347854845137454d00/,
     &     r4b/0.339981043584856d00/,w4b/0.652145154862546d00/
c
      if (nints.eq.1) then
         raone(1) = zero
      endif
c
      if(nnods.eq.1) then
         xaone(1) = zero
         paone(1) = zero
      end if
c
      if (nints.eq.2) then
         raone(1)=-.577350269189626
         raone(2)= .577350269189625
c
         paone(1)= .577350269189626
         paone(2)=-.577350269189625
c
         taone(1)= .577350269189626
         taone(2)=-.577350269189625
      endif
c
      if (nnods.eq.2) then
         xaone(1) = -one
         xaone(2) =  one
      endif
c
      if (nints.eq.3) then
         raone(1)=-.774596669241483
         raone(2)= .774596669241483
         raone(3)= zero
c
         paone(1)= .774596669241483
         paone(2)=-.774596669241483
         paone(3)= zero
c
         taone(1)= .774596669241483
         taone(2)=-.774596669241483
         taone(3)= zero
      endif
c
      if(nnods.eq.3) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
      endif
c
      if(nen.eq.1) inod(1,1,1) = 1
c
      if(nen.eq.8) then
         inod(1,1,1) = 1
         inod(2,1,1) = 2
         inod(1,2,1) = 4
         inod(2,2,1) = 3
c
         inod(1,1,2) = 5
         inod(2,1,2) = 6
         inod(1,2,2) = 8
         inod(2,2,2) = 7
      end if
c
c     build shape functions
c
      write(*,*) "BUILDING SHP FUNC"
      lb=0
      do ns=1,nside
         do l = 1, nints
            lb = lb + 1
c     frente
            if(ns.eq.1) then
               r = raone(l)
               s = -one
               t = taone(l)
            end if
c     esq
            if(ns.eq.2) then
               r = -one
               s = paone(l)
               t = taone(l)
            end if
c     dir
            if(ns.eq.3) then
               r = one
               s = raone(l)
               t = taone(l)
            end if
c     tras
            if(ns.eq.4) then
               r = raone(l)
               s = one
               t = taone(l)
            end if
c     baixo
            if(ns.eq.5) then
               r = raone(l)
               s = paone(l)
               t = -one
            end if
c     cima
            if(ns.eq.6) then
               r = raone(l)
               s = paone(l)
               t = one
            end if
c
            shlx(1,1) = zero
            shly(1,1) = zero
            shlz(1,1) = zero
            shlx(2,1) = one
            shly(2,1) = one
            shlz(2,1) = one
            if(nnods.eq.1) go to 100
c
            do i = 1, nnods
               aa = one
               bb = one
               cc = one
               dd = one
               aax = zero
               aay = zero
               aaz = zero
               do j = 1, nnods
                  daj = one
                  caj = one
                  baj = one
                  if (i.ne.j) then
                     aa = aa * (r - xaone(j))
                     cc = cc * (s - xaone(j))
                     dd = dd * (t - xaone(j))
                     bb = bb * (xaone(i) - xaone(j))
                     do k = 1, nnods
                        if(k.ne.i.and.k.ne.j) daj=daj*(r-xaone(k))
                        if(k.ne.i.and.k.ne.j) caj=caj*(s-xaone(k))
                        if(k.ne.i.and.k.ne.j) baj=baj*(t-xaone(k))
                     end do
                     aax = aax + daj
                     aay = aay + caj
                     aaz = aaz + baj
                  endif
               end do
c
               shlx(1,i) = aax/bb
               shly(1,i) = aay/bb
               shlz(1,i) = aaz/bb
c
               shlx(2,i) = aa/bb
               shly(2,i) = cc/bb
               shlz(2,i) = dd/bb
            end do
c
 100        continue
c
            do iz=1,nnods
               do iy=1,nnods
                  do ix=1,nnods
                     j = inod(ix,iy,iz)
                     shl(1,j,lb) = shlx(1,ix)*shly(2,iy)*shlz(2,iz)
                     shl(2,j,lb) = shlx(2,ix)*shly(1,iy)*shlz(2,iz)
                     shl(3,j,lb) = shlx(2,ix)*shly(2,iy)*shlz(1,iz)
                     shl(4,j,lb) = shlx(2,ix)*shly(2,iy)*shlz(2,iz)
                  end do
               end do
            end do
c
         end do
      end do
c
      return
      end


c-----------------------------------------------------------------------
      subroutine flninter(ien   ,x     ,xl   ,
     &                 d     ,dl    ,mat   ,
     &                 c     ,ipar  ,dlf   ,
     &                 dlp   ,dsfl  ,det   ,
     &                 shl   , shg  ,wt    ,
     &                 detc  ,shlc  , shgc ,
     &                 ddis  ,detp  ,shlp  ,
     &                 shgp ,
c
     &                 shln  ,shgn  ,
     &                 detn  ,shlb  ,shgb  ,
     &                 detpn ,shlpn ,shgpn ,
     &                 idside,xls   ,idlsd ,
     &                 grav  ,wn    ,
c
     &                 numel ,neesq ,nen   ,nsd   ,
     &                 nesd  ,nint  ,neg   ,nrowsh,
     &                 ned   ,nee   ,numnp ,ndof  ,
     &                 ncon  ,nencon,necon ,index ,
     &                 nints,iplt ,nenp   ,
     &                 nside,nnods ,nenlad,npars  ,
     &                 nmultp, nodsp)
c-----------------------------------------------------------------------
c     Some variables used in this subroutine
c            j: degree of freedom (1,...ncon)
c         u(j): finite element solution
c        du(j): derivative of finite element solution
c        ue(j): exact solution
c       due(j): derivative of exact solution
c       el2(j): error in L2
c      epri(j): error in the seminorm of H1 (L2 of derivatives)
c     el2el(j): error in L2 in the element domain
c    epriel(j): error in the seminorm of H1 in the element domain
c-----------------------------------------------------------------------
c    Program to calculate and print the L2 and H1 seminorm of the error
c    for each degree of freedom in the finite element solution .
c    The trapezoidal rule is used for integration in each element .
c    The number of integration points is given by nints .
c
c    This version is only applicable to 2D.
c
      implicit real*8 (a-h,o-z)
c
c     remove above card for single-precision operation
c
      character *1 tab
      dimension ien(nen,*),x(nsd,*),xl(nesd,*),d(ndof,*),dl(ned,*),
     &          mat(*),c(10,*),ipar(nodsp,*),dlf(ncon,*) ,
     &          dsfl(ncon,nencon,*),ddis(ned,nenp,*)
      dimension u(6),dux(6),duy(6),ue(6),duex(6),duey(6)
      dimension shl(3,nen,*),shg(3,nen,*),det(*),wt(*)
      dimension shlc(3,nencon,*),shgc(3,nencon,*),detc(*)
      dimension el2(6),eprix(6),epriy(6),el2el(6),
     &          eprxel(6),epryel(6)
	    dimension detp(*),shlp(3,nenp,*),shgp(3,nenp,*),dlp(ned,*)
c
      dimension detn(*),shlb(3,nenlad,*),shgb(3,nenlad,*),
     &          detpn(*),shlpn(3,npars,*),shgpn(3,npars,*),
     &          idside(nside,*),xls(nesd,*),idlsd(*)
	    dimension dls(12),grav(*),wn(*)
      dimension shln(3,nnods,*),shgn(3,nnods,*)
      dimension sxlhp(64,64),sxlhv(64,64),xlpn(2,64),xlvn(2,64)
      dimension shlsd(8,8),xlps(2,8)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      pi=4.*datan(1.d00)
      dpi=2.d00*pi
c
      gf1=grav(1)
      gf2=grav(2)
      gf3=grav(3)
      eps=c(1,1)
c
         tab=char(9)
         xint=0.d00
         yint=0.d00
c
      call clear ( el2, ncon )
      call clear (eprix, ncon )
      call clear (epriy, ncon )
c
      divu2=0.d00
	edp = 0.d00
	edpx= 0.d00
	edpy= 0.d00
	xmlt= 0.d00
c
c
      call shapesd(shlsd,nenlad,npars)
      if (nen.eq.3) then
       call shapent(sxlhp,nen,nenp)
       call shapent(sxlhv,nen,nencon)
      else if (nen.eq.4) then
        call shapen(sxlhp,nen,nenp)
        call shapen(sxlhv,nen,nencon)
      else
        stop
      endif
c
cc       call shapesd(shlsd,nenlad,npars)
cc       call shapen(sxlhp,nen,nenp)
cc       call shapen(sxlhv,nen,nencon)
c
c    .  loop on elements
c
      do 50 n=1,numel
c
       call local(ien(1,n),x,xl,nen,nsd,nesd)
       call local(ipar(1,n),d,dl,nodsp,ndof,ned)
c
      do i=1,nenp
      xlpn(1,i) = 0.d00
      xlpn(2,i) = 0.d00
      do j=1,nen
        xlpn(1,i) = xlpn(1,i) + sxlhp(j,i)*xl(1,j)
        xlpn(2,i) = xlpn(2,i) + sxlhp(j,i)*xl(2,j)
       end do
       end do
c
       call elemdlp(xlpn,dlp,eps,ned,nenp,index)
c
      do i=1,nencon
      xlvn(1,i) = 0.d00
      xlvn(2,i) = 0.d00
      do j=1,nen
        xlvn(1,i) = xlvn(1,i) + sxlhv(j,i)*xl(1,j)
        xlvn(2,i) = xlvn(2,i) + sxlhv(j,i)*xl(2,j)
       end do
       end do
c
       call elemdlf(xlvn,dlf,eps,ncon,nencon,index)
c
      call clear ( el2el, ncon )
      call clear (eprxel, ncon )
      call clear (epryel, ncon )
      divue=0.d00
	pmede=0.d00
	dpe = 0.d00
	dpex= 0.d00
	dpey= 0.d00
c
c    .  loop on integration points
c
c
c    .Triangles or quadrilaterals
c
c
      call shgqs(xl,detc,shlc,shgc,nint,n,neg,.true.,nencon,shl,nen)
c
      call shgqs(xl,detp,shlp,shgp,nint,n,neg,.true.,nenp,shl,nen)
c
c
        do 4040 l=1,nint
        ct=detc(l)*wt(l)
        call clear ( u, ncon )
        call clear (dux, ncon )
        call clear (duy, ncon )
        xint=0.d0
        yint=0.d0
         do 3030 i=1,nen
            xint=xint+shl(3,i,l)*xl(1,i)
            yint=yint+shl(3,i,l)*xl(2,i)
3030     continue
c
          do 3330 i=1,nencon
            do 2020 j=1,ncon
                  u(j)   = u(j)   + shgc(3,i,l)*dlf(j,i)
	          dux(j) = dux(j) + shgc(1,i,l)*dlf(j,i)
	          duy(j) = duy(j) + shgc(2,i,l)*dlf(j,i)
2020        continue
3330      continue
c
         call uexafx(xint,yint,ue,duex,duey,eps,index)
c
c     pressao descontinua
c
        pe = 0.d00
	pex = 0.d00
	pey = 0.d00
	do i=1,nenp
	  pe = pe + shgp(3,i,l)*dlp(1,i)
	  pex= pex+ shgp(1,i,l)*dlp(1,i)
	  pey= pey+ shgp(2,i,l)*dlp(1,i)
	end do
	  dpe = dpe + ct*(pe - ue(3))**2
	  dpex= dpex + ct*(pex-duex(3))**2
	  dpey= dpey + ct*(pey-duey(3))**2
c
      if(iwrite.ne.0) then
         write(43,2424) n,l,u(1),ue(1),u(2),ue(2),pe,ue(3)
2424   format(2i10,6e15.5)
       end if
c
       do 3535 j=1,ncon
        un = ct * ( (u(j)-ue(j))**2 )
        upnx= ct * ( (dux(j)-duex(j))**2 )
        upny= ct * ( (duy(j)-duey(j))**2 )
        el2el(j) = el2el(j) + un
        eprxel(j) = eprxel(j) + upnx
        epryel(j) = epryel(j) + upny
3535   continue
c
c     conservacao
c
      divue = divue + (dux(1) + duy(2) - duex(1) - duey(2))*ct
     &              + eps*(pe-ue(3))*ct     !apagar conservacao
4040  continue
c
       do 45 j=1,ncon
        el2(j) = el2(j) + el2el(j)
        eprix(j) = eprix(j) + eprxel(j)
        epriy(j) = epriy(j) + epryel(j)
  45   continue
c
      divu2 = divu2 + divue**2
	edp = edp + dpe
	edpx= edpx+ dpex
	edpy= edpy+ dpey
c
c    ..boundary terms - multiplier
c
	xmlte = 0.d00
      do 4000 ns=1,nside
c
c-----localiza os parametros do lado n
c
      do nn=1,npars
	  nld = (ns-1)*npars + nn
	  dls(nn) = dl(1,nld)
cc	  write(37,*) nn,dls(nn),'inter'
      end do
c
c-----localiza os no's do lado ns
c
c
      ns1=idside(ns,1)
      ns2=idside(ns,2)
      nl1=ien(ns1,n)
      nl2=ien(ns2,n)
c
      if(nl2.gt.nl1) then
	  sign = 1.d00
        do  nn=1,nenlad
           idlsd(nn)=idside(ns,nn)
	  end do
	else
	  sign = -1.d00
	  idlsd(1) = idside(ns,2)
	  idlsd(2) = idside(ns,1)
	  id3 = nenlad-2
	  if(id3.gt.0) then
	    do il=id3,nenlad
	      idlsd(il) = idside(ns,nenlad+id3-il)
	    end do
	  end if
      end if
c
c
      do 2000 nn=1,nenlad
      nl=idlsd(nn)
      xls(1,nn)=xl(1,nl)
      xls(2,nn)=xl(2,nl)
c
 2000 continue
c
      do nn=1,npars
      xlps(1,nn) = 0.d00
      xlps(2,nn) = 0.d00
      do jj=1,nenlad
        xlps(1,nn) = xlps(1,nn) + shlsd(jj,nn)*xls(1,jj)
        xlps(2,nn) = xlps(2,nn) + shlsd(jj,nn)*xls(2,jj)
      end do
      end do
c
cc       call elemult(xlps,dls,eps,ned,npars,index)
c
      call oneshgp(xls,detn,shlb,shln,shgn,
     &             nenlad,nnods,nints,nesd,ns,n,neg)
c
      call oneshgp(xls,detpn,shlb,shlpn,shgpn,
     &             nenlad,npars,nints,nesd,ns,n,neg)
c
c
c    .compute boundary integral
c

      do 1000 ls=1,nints
c
      cwn = wn(ls)*detn(ls)
c
c    valores dos parametros do multiplicador
c
	  dhs=0.d00
        do i=1,npars
	    dhs = dhs + dls(i)*shgpn(2,i,ls)
        end do
c
c     geometria
c
        x1 =0.d00
        x2 =0.d00
        dx1=0.d00
        dx2=0.d00
c
        do i=1,nenlad
          x1 =x1 +xls(1,i)*shlb(2,i,ls)
          x2 =x2 +xls(2,i)*shlb(2,i,ls)
          dx1=dx1+xls(1,i)*shlb(1,i,ls)
          dx2=dx2+xls(2,i)*shlb(1,i,ls)
        end do
           dxx=dsqrt(dx1*dx1+dx2*dx2)
           xn1= sign*dx2/dxx
           xn2=-sign*dx1/dxx
c
c    valores exatos dos multiplicadores
c
c
      pix=pi*x1
      piy=pi*x2
      sx=dsin(pix)
      sy=dsin(piy)
      cx=dcos(pix)
      cy=dcos(piy)
c
      pi2=pi*pi
      xinvpes = eps**(-1.0)
      co=1.d00
c
c    valor exato do multiplicador
c
      dhse= gf1*sx*sy
     &    + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*(1.d00 - exp((x1
     &          - 1.d00)/eps))*(1.d00 - exp((x2 - 1.d00)/eps)))
     &    + gf3*1.d00
c
      xmlte = xmlte + cwn*(dhse-dhs)**2
c
 1000 continue
 4000 continue
c
      xmlt = xmlt + xmlte
c
  50   continue
c
      divu = dsqrt(divu2)
c
c    .. error in all domain
c
       fxl2 = dsqrt(el2(1) + el2(2))
       fxh1 = dsqrt(eprix(1)+eprix(2) + epriy(1)+epriy(2))

	edp = dlog10(dsqrt(edp))
	gradp= dlog10(dsqrt(edpx+edpy))

       fxl2 = dlog10(fxl2)
       fxh1 = dlog10(fxh1)
	 xmlt = dlog10(dsqrt(xmlt))
       xel = dfloat(numel)/dfloat(nnods-1)**2
       xnp = nmultp
       xel = dlog10(xel)/2.d00
       xnp = dlog10(xnp)/2.d00

      write(iplt,2101) xel,xnp,fxl2,fxh1,edp,gradp,xmlt,divu
      write(999,2101)  xel,xnp,eprix(1),eprix(2),epriy(1),epriy(2)
       return
c
2101  format(9(e13.4))
      end

c-----------------------------------------------------------------------
      subroutine flnormp(ien   ,x     ,xl   ,
     &                   d     ,dl    ,mat   ,
     &                   c     ,ipar  ,dlf   ,
     &                   dlp   ,dsfl  ,det   ,
     &                   shl   ,shg  ,wt    ,
     &                   detc  ,shlc  , shgc ,
     &                   ddis  ,detp  ,shlp  ,
     &                   shgp ,
c
     &                   shln  ,shgn  ,
     &                   detn  ,shlb  ,shgb  ,
     &                   detpn ,shlpn ,shgpn ,
     &                   idside,xls   ,idlsd ,
     &                   grav  ,wn    ,
c
     &                   numel ,neesq ,nen   ,nsd   ,
     &                   nesd  ,nint  ,neg   ,nrowsh,
     &                   ned   ,nee   ,numnp ,ndof  ,
     &                   ncon  ,nencon,necon ,index ,
     &                   nints, iplt  ,nenp  ,
     &                   nside, nnods,nenlad ,npars  ,
     &                   nmultp, nodsp)
c------------------------------------------------------------------
c     Some variables used in this subroutine
c            j: degree of freedom (1,...ncon)
c         u(j): finite element solution
c        du(j): derivative of finite element solution
c        ue(j): exact solution
c       due(j): derivative of exact solution
c       el2(j): error in L2
c      epri(j): error in the seminorm of H1 (L2 of derivatives)
c     el2el(j): error in L2 in the element domain
c    epriel(j): error in the seminorm of H1 in the element domain
c----------------------------------------------------------------
c    Program to calculate and print the L2 and H1 seminorm of the error
c    for each degree of freedom in the finite element solution .
c    The trapezoidal rule is used for integration in each element .
c    The number of integration points is given by nints .
c
c     This version is only applicable to 2D/3D
c     Writes to:
c        erro-local-primal-shdg-dc.dat
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      character *1 tab
      dimension ien(nen,*),x(nsd,*),xl(nesd,*)
      dimension d(ndof,*),dl(ned,*)
      dimension mat(*),c(10,*),ipar(nodsp,*),dlf(ncon,*)
      dimension dsfl(ncon,nencon,*),ddis(ned,nenp,*)
      dimension u(6),dux(6),duy(6),duz(6)
      dimension ue(6),duex(6),duey(6),duez(6)
      dimension el2(6),eprix(6),epriy(6),epriz(6)
      dimension el2el(6),eprxel(6),epryel(6),eprzel(6)
c
      dimension shl (nrowsh+1,nen,*),   shg(nrowsh+1,nen,*)
      dimension shlp(nrowsh+1,nenp,*),  shgp(nrowsh+1,nenp,*)
      dimension shlc(nrowsh+1,nencon,*),shgc(nrowsh+1,nencon,*)
      dimension det(*),detn(*),detp(*),wt(*),wn(*)
c
      dimension shlb (nrowsh,nenlad,*),shgb(nrowsh,nenlad,*)
      dimension shln (nrowsh,nnods,*), shgn(nrowsh,nnods,*)
      dimension shlpn(nrowsh,npars,*),shgpn(nrowsh,npars,*)
      dimension detc(*),detpn(*)
c
      dimension dlp(ned,*), dls(12)
      dimension idside(nside,*),xls(nesd,*),idlsd(*),grav(*)
c
      dimension coo(3),vab(3),vad(3),vc(3),xn(3),xxn(3)
c
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
      common /iounit/ iin,ipp,ipmx,ieco,ilp,ilocal,interpl,ielmat,iwrite
c
      write(*,*) "DEBUG 1"
      pi  = 4.*datan(1.d00)
      dpi = 2.d00*pi
c
      gf1 = grav(1)
      gf2 = grav(2)
      gf3 = grav(3)
      eps = c(1,1)
c
      tab = char(9)
      xint = 0.d00
      yint = 0.d00
      zint = 0.d00
c
      call clear ( el2,  ncon )
      call clear (eprix, ncon )
      call clear (epriy, ncon )
      call clear (epriz, ncon )
c
      divu2 = 0.d00
      edp   = 0.d00
      edpx  = 0.d00
      edpy  = 0.d00
      edpz  = 0.d00
      xmlt  = 0.d00
c
c     debug
c
c      write(*,*) "MULTIPLICADORES"
c      do kk=1,nmultp
c         write(*,'(I5,F10.6)') kk, d(1,kk)
c         if(mod(kk,4).eq.0) then
c            write(*,*) ""
c         end if
c      end do

      write(*,*) "DEBUG 2"
c
c     loop on elements
c
      do 50 n=1,numel
         write(*,*) " "
         write(*,'(A,I3)') "elemento ", n
c
         write(*,*) "DEBUG 3"
         call local(ien(1,n),x,xl,nen,nsd,nesd)
c
c     DEBUG LOCAL
c         write(*,*) "copia de D para DL (nos indices dados por IPAR)"
c         write(*,*) nodsp, ndof, ned
c
         write(*,*) "DEBUG 4",nodsp,ndof,ned
         call local(ipar(1,n),d,dl,nodsp,ndof,ned)
c
         write(*,*) "DEBUG 5"
c        write(*,'(24I4)') (ipar(kk,n),kk=1,24)
         write(*,'(A)') " array dl"
         do kk=1,6
            write(*,'(24F8.4)') (dl(1,4*(kk-1)+jj),jj=1,4)
         end do
         write(*,*) "DEBUG 6"
         write(*,'(A)') " elem nodes"
         do kk=1,nen
            write(*,'(3F8.4)') xl(1,kk), xl(2,kk), xl(3,kk)
         end do
         write(*,*) "DEBUG 7"
c
c     recupera multiplicador
c
         do i=1,nenp
            dlp(1,i) = ddis(1,i,n)
         end do
c
c     calcula centroide
c
         call centroid(xl,nesd,nen,coo)
c
         call clear( el2el, ncon )
         call clear(eprxel, ncon )
         call clear(epryel, ncon )
c
         divue = 0.d00
         pmede = 0.d00
         dpe    = 0.d00
         dpex   = 0.d00
         dpey   = 0.d00
         dpez   = 0.d00
c
c     2D: triangles or quads
c     3D: hexahedron
c
         if(nsd.eq.2) then
          call shgqs(xl,detc,shlc,shgc,nint,n,neg,.true.,nencon,shl,nen)
          call shgqs(xl,detp,shlp,shgp,nint,n,neg,.true.,nenp,shl,nen)
         else if(nsd.eq.3) then
          call shghx(xl,detc,shlc,shgc,nint,n,neg,.true.,nencon,shl,nen)
          call shghx(xl,detp,shlp,shgp,nint,n,neg,.true.,nenp,shl,nen)
         end if
c
c     loop on integration points (volume)
c
         write(*,'(A)') " exata  / variavel / erro no pt integracao"
         do l=1,nint
            ct=detc(l)*wt(l)
            call clear ( u, ncon )
            call clear (dux, ncon )
            call clear (duy, ncon )
            call clear (duz, ncon )
            xint = 0.d0
            yint = 0.d0
            zint = 0.d0
            do i=1,nen
               xint = xint + shl(4,i,l)*xl(1,i)
               yint = yint + shl(4,i,l)*xl(2,i)
               zint = zint + shl(4,i,l)*xl(3,i)
            end do
c
c     calculo da solucao exata
c
c           call uexafx(xint,yint,ue,duex,duey,eps,index)
            call uexafx3(xint,yint,zint,ue,duex,duey,duez,eps,index)
c
c     pressao descontinua
c
            pe  = 0.d00
            pex = 0.d00
            pey = 0.d00
            pez = 0.d00
c
            do i=1,nenp
               pe  = pe  + shgp(4,i,l)*dlp(1,i)
               pex = pex + shgp(1,i,l)*dlp(1,i)
               pey = pey + shgp(2,i,l)*dlp(1,i)
               pez = pez + shgp(3,i,l)*dlp(1,i)
            end do
c
            dpe  = dpe  + ct*(pe - ue(3))**2
            dpex = dpex + ct*(pex-duex(3))**2
            dpey = dpey + ct*(pey-duey(3))**2
            dpez = dpez + ct*(pez-duez(3))**2
c
            if(iwrite.ne.0) then
               write(43,2424) n,l,u(1),ue(1),u(2),ue(2),pe,ue(3)
 2424          format(2i10,6e15.5)
            end if
c
c     DEBUG
c            write(*,'(A,8F10.6)') "dlp       ",(dlp(1,kk),kk=1,nenp)
c            write(*,'(A,4F10.6)') "ponto exa ",xint,yint,zint,ue(3)
            write(*,'(6F8.4)') ue(3),pe,dabs(pe-ue(3))

c
         end do
c
         do j=1,ncon
            el2(j) = el2(j) + el2el(j)
            eprix(j) = eprix(j) + eprxel(j)
            epriy(j) = epriy(j) + epryel(j)
         end do
c
         divu2 = divu2 + divue**2
c
         edp  = edp  + dpe
         edpx = edpx + dpex
         edpy = edpy + dpey
         edpz = edpz + dpez
c
c     boundary terms - multiplier
c
         xmlte = 0.d00
         do 4000 ns=1,nside
            write(*,*) " "
            write(*,*) "face ", ns
c
c     localiza os parametros do lado ns
c
            do nn=1,npars
               nld = (ns-1)*npars + nn
               dls(nn) = dl(1,nld)
c               write(*,*) "copia do DL de ", nld, " para DLS em ", nn
c               write(*,'(I5)') nld
            end do
c
c     localiza os no's do lado ns
c
            ns1 = idside(ns,1)
            ns2 = idside(ns,2)
            ns3 = idside(ns,3)
            ns4 = idside(ns,4)
c
c     n=nel (numero do elemento)
c
            nl1 = ien(ns1,n)
            nl2 = ien(ns2,n)
            nl3 = ien(ns3,n)
            nl4 = ien(ns4,n)
            write(*,'(A,4I5)') " local  ",ns1,ns2,ns3,ns4
            write(*,'(A,4I5)') " global ",nl1,nl2,nl3,nl4
            write(*,'(A)') " dls "
            write(*,'(10F8.4)') (dls(nn),nn=1,npars)
c            write(*,'(4F8.4)') (xl(kd,ns1), kd=1,3)
c            write(*,'(4F8.4)') (xl(kd,ns2), kd=1,3)
c            write(*,'(4F8.4)') (xl(kd,ns3), kd=1,3)
c            write(*,'(4F8.4)') (xl(kd,ns4), kd=1,3)

c
c     calcula sign para a aresta/face
c
            call vec3sub(xl(1,ns2), xl(1,ns1), vab)
            call vec3sub(xl(1,ns4), xl(1,ns1), vad)
            call vec3sub(coo,xl(1,ns1),vc)
            call cross(vab,vad,xn)
            call nrm3(xn)
            dotcn = dot3(vc,xn)
c
            if(dotcn.lt.0.d0) then
               sign = 1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
            else
               write(*,'(A)') " trocou sinal"
               sign = -1.d00
               do nn=1,nenlad
                  idlsd(nn) = idside(ns,nn)
               end do
c
c     TODO: nao precisa fazer troca aqui
c
c               idlsd(1) = idside(ns,4)
c               idlsd(2) = idside(ns,3)
c               idlsd(3) = idside(ns,2)
c               idlsd(4) = idside(ns,1)
            end if
c
c     muda a normal (se necessario)
c
            xn(1)=sign*xn(1)
            xn(2)=sign*xn(2)
            xn(3)=sign*xn(3)
c
            do nn=1,nenlad
               nl=idlsd(nn)
               xls(1,nn) = xl(1,nl)
               xls(2,nn) = xl(2,nl)
               xls(3,nn) = xl(3,nl)
            end do

c
c     calculo funcoes globais
c
            if(nesd.eq.2) then
c
c     PROBLEMA 2D
c
               call oneshgp(xls,detn,shlb,shln,shgn,
     &              nenlad,nnods,nints,nesd,ns,n,neg)
c
               call oneshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad,npars,nints,nesd,ns,n,neg)
            else
c
c     PROBLEMA 3D
c
c               write(*,*) "computing twoshgp (flnormp)"
               nenlad2=4
               npars2=npars
               nnods2=4
ccc               call twoshgp(xls,detn,shlb,shln,shgn,
ccc     &              nenlad2,nnods2,nints,nesd,ns,n,neg,xxn)
               call twoshgp(xls,detpn,shlb,shlpn,shgpn,
     &              nenlad2,npars2,nints,nesd,ns,n,neg,xxn)
            end if
c
c     compute boundary integral
c
            do ls=1,nints
               cwn = wn(ls)*detpn(ls)
c
c     valores dos parametros do multiplicador
c
               dhs = 0.d00
               do i=1,npars
                  dhs = dhs + dls(i)*shgpn(3,i,ls)
               end do
c              write(*,'(A)') " interpolando multiplicador na face"
c              write(*,'(10F8.4)') (shgpn(3,i,ls), i=1,npars)
c              write(*,'(F8.4)') dhs

c
c     geometria
c
               x1 = 0.d00
               x2 = 0.d00
               x3 = 0.d00
               dx1 = 0.d00
               dx2 = 0.d00
               dx3 = 0.d00
c
               do i=1,nenlad
                  x1 = x1 + xls(1,i)*shlb(3,i,ls)
                  x2 = x2 + xls(2,i)*shlb(3,i,ls)
                  x3 = x3 + xls(3,i)*shlb(3,i,ls)
c                  dx1 = dx1 + xls(1,i)*shlb(1,i,ls)
c                  dx2 = dx2 + xls(2,i)*shlb(1,i,ls)
c                  dx3 = dx3 + xls(3,i)*shlb(1,i,ls)
               end do

c               write(*,*) "SHLB FLNORMP"
c               write(*,'(10F8.4)') (shlb(3,i,ls), i=1,nenlad)
c               write(*,*) "INTERPOLOU PONTOS DE INT p/ FACE DO MULTP"
c               write(*,*) x1,x2,x3

c               dxx = dsqrt(dx1*dx1+dx2*dx2)
c               xn1 = sign*dx2/dxx
c               xn2 =-sign*dx1/dxx

c
c     normal da face
c
               xn1 = xn(1)
               xn2 = xn(2)
               xn3 = xn(3)
c               write(*,*) "NORMAL XXN", xn1, xn2, xn3
c               write(*,*) "NORMAL XN", xn(1), xn(2), xn(3)
c              write(*,*) "VECDLS ", (dls(kk),kk=1,4)
c
c     valores exatos dos multiplicadores
c
               pix=pi*x1
               piy=pi*x2
               piz=pi*x3
               sx=dsin(pix)
               sy=dsin(piy)
               sz=dsin(piz)
               cx=dcos(pix)
               cy=dcos(piy)
               cz=dcos(piz)
c
               pi2=pi*pi
               xinvpes = eps**(-1.0)
               co=1.d00
c
c     valor exato do multiplicador
c
               dhse = gf1*sx*sy*sz
     &              + gf2*(dsin(pix/2.d00)*dsin(piy/2.d00)*
     &              (1.d00 - exp((x1-1.d00)/eps))*
     &              (1.d00 - exp((x2 - 1.d00)/eps))
     &              ) + gf3*1.d00
c
               xmlte = xmlte + cwn*(dhse-dhs)**2
c
               if(ls.eq.1) then
                  write(*,'(A)',advance='no') "   x   /   y    /    z "
                  write(*,'(A)') " / exato /  aprox  / erro"
               end if
               write(*,'(6F8.4)') x1,x2,x3,dhse,dhs,dabs(dhse-dhs)
            end do
 4000    continue
c     fim do loop nas faces/arestas

c
         xmlt = xmlt + xmlte
c
 50   continue
c
c     error in all domain
c     edp   -> log p
c     gradp -> log grad p
c     xmlt  -> log mult (multiplicador)
c
      edp   = dlog10(dsqrt(edp))
      gradp = dlog10(dsqrt(edpx+edpy))
      xmlt  = dlog10(dsqrt(xmlt))
c
c     log(h)
c
      xel = dfloat(numel)/dfloat(nnods-1)**3
      xel = dlog10(xel)/2.d00
c
c     log(k)
c
      xnp = nmultp
      xnp = dlog10(xnp)/2.d00

      write(iplt,2101) xel,xnp,edp,gradp,xmlt
      return
c
 2101 format(9(e13.4))
      end

c-----------------------------------------------------------------------
      subroutine shapent(shl,nen,nenc)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension shl(64,64),cl1(64),cl2(64),cl3(64)
      data   zero,pt1667,pt25,pt5
     &      /0.0d0,0.1666666666666667d0,0.25d0,0.5d0/,
     &       one,two,three,four,five,six
     &      /1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0/
      data r1/0.33333333333333333333d00/ ,w1/1.d00/,
     &     r2/0.5d00                   /,w2/0.3333333333333333333d00/
c
      pt1s3 = one/three
      pt2s3 = two/three
      pt1s4 = one/four
      pt2s4 = two/four
      pt3s4 = three/four
      pt1s5 = one/five
      pt2s5 = two/five
      pt3s5 = three/five
      pt4s5 = four/five
c
      pt3s2 = three/two
      pt9s2 = 9.d0/two
      pt27s2= 27.d0/two
      pt2s3 = two/three
      pt1s6 = one/six
      pt32s3= 32.d0/three
      pt8s3 = 8.d0/three

c
      if (nenc.eq.1) then
            cl1(1)=r1
            cl2(1)=r1
            cl3(1)=one-r1-r1
      end if
c
      if(nenc.eq.3) then
        cl1(1)=one
        cl2(1)=zero
        cl3(1)=zero

        cl1(2)=zero
        cl2(2)=one
        cl3(2)=zero

        cl1(3)=zero
        cl2(3)=zero
        cl3(3)=one
      end if
c
      if(nenc.eq.6) then
        cl1(1)=one
        cl2(1)=zero
        cl3(1)=zero

        cl1(2)=zero
        cl2(2)=one
        cl3(2)=zero

        cl1(3)=zero
        cl2(3)=zero
        cl3(3)=one

        cl1(4)=r2
        cl2(4)=r2
        cl3(4)=zero

        cl1(5)=zero
        cl2(5)=r2
        cl3(5)=r2

        cl1(6)=r2
        cl2(6)=zero
        cl3(6)=r2

      end if
c
      if(nenc.eq.10) then
        cl1(1)=one
        cl2(1)=zero
        cl3(1)=zero

        cl1(2)=zero
        cl2(2)=one
        cl3(2)=zero

        cl1(3)=zero
        cl2(3)=zero
        cl3(3)=one

        cl1(4)=pt2s3
        cl2(4)=pt1s3
        cl3(4)=zero

        cl1(5)=pt1s3
        cl2(5)=pt2s3
        cl3(5)=zero

        cl1(6)=zero
        cl2(6)=pt2s3
        cl3(6)=pt1s3

        cl1(7)=zero
        cl2(7)=pt1s3
        cl3(7)=pt2s3

        cl1(8)=pt1s3
        cl2(8)=zero
        cl3(8)=pt2s3

        cl1(9)=pt2s3
        cl2(9)=zero
        cl3(9)=pt1s3

        cl1(10)=r1
        cl2(10)=r1
        cl3(10)=r1

      end if
c
      if(nenc.eq.15) then
        cl1(1)=one
        cl2(1)=zero
        cl3(1)=zero

        cl1(2)=zero
        cl2(2)=one
        cl3(2)=zero

        cl1(3)=zero
        cl2(3)=zero
        cl3(3)=one

        cl1(4)=pt3s4
        cl2(4)=pt1s4
        cl3(4)=zero

        cl1(5)=pt2s4
        cl2(5)=pt2s4
        cl3(5)=zero

        cl1(6)=pt1s4
        cl2(6)=pt3s4
        cl3(6)=zero

        cl1(7)=zero
        cl2(7)=pt3s4
        cl3(7)=pt1s4

        cl1(8)=zero
        cl2(8)=pt2s4
        cl3(8)=pt2s4

        cl1(9)=zero
        cl2(9)=pt1s4
        cl3(9)=pt3s4

        cl1(10)=pt1s4
        cl2(10)=zero
        cl3(10)=pt3s4

        cl1(11)=pt2s4
        cl2(11)=zero
        cl3(11)=pt2s4

        cl1(12)=pt3s4
        cl2(12)=zero
        cl3(12)=pt1s4

        cl1(13)=pt2s4
        cl2(13)=pt1s4
        cl3(13)=pt1s4

        cl1(14)=pt1s4
        cl2(14)=pt2s4
        cl3(14)=pt1s4

        cl1(15)=pt1s4
        cl2(15)=pt1s4
        cl3(15)=pt2s4
      end if
c
      if(nenc.eq.21) then
        cl1(1)=one
        cl2(1)=zero
        cl3(1)=zero

        cl1(2)=zero
        cl2(2)=one
        cl3(2)=zero

        cl1(3)=zero
        cl2(3)=zero
        cl3(3)=one

        cl1(4)=pt4s5
        cl2(4)=pt1s5
        cl3(4)=zero

        cl1(5)=pt3s5
        cl2(5)=pt2s5
        cl3(5)=zero

        cl1(6)=pt2s5
        cl2(6)=pt3s5
        cl3(6)=zero

        cl1(7)=pt1s5
        cl2(7)=pt4s5
        cl3(7)=zero

        cl1(8)=zero
        cl2(8)=pt4s5
        cl3(8)=pt1s5

        cl1(9)=zero
        cl2(9)=pt3s5
        cl3(9)=pt2s5

        cl1(10)=zero
        cl2(10)=pt2s5
        cl3(10)=pt3s5

        cl1(11)=zero
        cl2(11)=pt1s5
        cl3(11)=pt4s5

        cl1(12)=pt1s5
        cl2(12)=zero
        cl3(12)=pt4s5

        cl1(13)=pt2s5
        cl2(13)=zero
        cl3(13)=pt3s5

        cl1(14)=pt3s5
        cl2(14)=zero
        cl3(14)=pt2s5

        cl1(15)=pt4s5
        cl2(15)=zero
        cl3(15)=pt1s5

        cl1(16)=pt3s5
        cl2(16)=pt1s5
        cl3(16)=pt1s5

        cl1(17)=pt2s5
        cl2(17)=pt2s5
        cl3(17)=pt1s5

        cl1(18)=pt1s5
        cl2(18)=pt3s5
        cl3(18)=pt1s5

        cl1(19)=pt1s5
        cl2(19)=pt2s5
        cl3(19)=pt2s5

        cl1(20)=pt1s5
        cl2(20)=pt1s5
        cl3(20)=pt3s5

        cl1(21)=pt2s5
        cl2(21)=pt1s5
        cl3(21)=pt2s5
      end if
c
      do 200 l=1,nenc
c
            c1 = cl1(l)
            c2 = cl2(l)
            c3 = cl3(l)
!          acrescentei o if (2012-10-08)
!           interpolação linear (p=1 e nen=3)(2012-10-08)
            if(nen.eq.3) then
        shl(1,l)= c1
        shl(2,l)= c2
        shl(3,l)= c3
            end if
!           interpolação quadrática (p=2 e nen=6)(2012-11-24)
            if(nen.eq.6) then
        shl(1,l)= (two*c1 - one)*c1
        shl(2,l)= (two*c2 - one)*c2
        shl(3,l)= (two*c3 - one)*c3
        shl(4,l)= four * c1 * c2
        shl(5,l)= four * c2 * c3
        shl(6,l)= four * c3 * c1
            end if
!           interpolação cúbica (p=3 e nen=10)(2012-11-24)
            if(nen.eq.10) then

        shl(1,l)= pt5*c1*(three*c1 - two)*(three*c1-one)

        shl(2,l)= pt5*c2*(three*c2 - two)*(three*c2-one)

        shl(3,l)= pt5*c3*(three*c3 - two)*(three*c3-one)

        shl(4,l)= pt9s2*c1*c2*(three*c1-one)

        shl(5,l)= pt9s2*c1*c2*(three*c2-one)

        shl(6,l)= pt9s2*c2*c3*(three*c2-one)

        shl(7,l)= pt9s2*c2*c3*(three*c3-one)

        shl(8,l)= pt9s2*c3*c1*(three*c3-one)

        shl(9,l)= pt9s2*c3*c1*(three*c1-one)

        shl(10,l)= 27.d0*c1*c2*c3

            end if
!           interpolação quártica (p=4 e nen=15)(2012-11-24)
            if(nen.eq.15) then

        shl(1,l)= pt1s6*(four*c1-three)*(four*c1-two)*
     &                  (four*c1-one)*c1

        shl(2,l)= pt1s6*(four*c2-three)*(four*c2-two)*
     &                  (four*c2-one)*c2

        shl(3,l)= pt1s6*(four*c3-three)*(four*c3-two)*
     &                   (four*c3-one)*c3

        shl(4,l)= pt8s3*c2*(four*c1-two)*(four*c1-one)*c1

        shl(5,l)= four*c2*(four*c2-one)*(four*c1-one)*c1

        shl(6,l)= pt8s3*c1*(four*c2-two)*(four*c2-one)*c2

        shl(7,l)= pt8s3*c3*(four*c2-two)*(four*c2-one)*c2

        shl(8,l)= four*(four*c2-one)*c2*(four*c3-one)*c3

        shl(9,l)= pt8s3*c2*(four*c3-two)*(four*c3-one)*c3

        shl(10,l)= pt8s3*c1*(four*c3-two)*(four*c3-one)*c3

        shl(11,l)= 4.d0*(four*c3-one)*c3*(four*c1-one)*c1

        shl(12,l)= pt8s3*c3*(four*c1-two)*(four*c1-one)*c1

        shl(13,l)= 32.d0*c1*c2*c3*(four*c1-one)

        shl(14,l)= 32.d0*c1*c2*c3*(four*c2-one)

        shl(15,l)= 32.d0*c1*c2*c3*(four*c3-one)

            end if
!           interpolação quíntica (p=5 e nen=21)(2012-11-24)
            if(nen.eq.21) then

        shl(1,l)= (one/24.d0)*(five*c1-four)*(five*c1-three)
     &                    *(five*c1-two)*(five*c1-one)*c1

        shl(2,l)= (one/24.d0)*(five*c2-four)*(five*c2-three)
     &                    *(five*c2-two)*(five*c2-one)*c2

        shl(3,l)= (one/24.d0)*(five*c3-four)*(five*c3-three)
     &                    *(five*c3-two)*(five*c3-one)*c3

        shl(4,l)= (25.d0/24.d0)*(five*c1-three)
     &                    *(five*c1-two)*(five*c1-one)*c1*c2

        shl(5,l)= (25.d0/12.d0)*(five*c1-two)
     &                    *(five*c1-one)*c1*(five*c2-one)*c2

        shl(6,l)= (25.d0/12.d0)*(five*c1-one)
     &                    *c1*(five*c2-two)*(five*c2-one)*c2

        shl(7,l)= (25.d0/24.d0)*c1*
     &                (five*c2-three)*(five*c2-two)*(five*c2-one)*c2

        shl(8,l)= (25.d0/24.d0)*c3*
     &                 (five*c2-three)*(five*c2-two)*(five*c2-one)*c2

        shl(9,l)= ((25.d0/12.d0)*(five*c3-one))
     &                     *c3*(five*c2-two)*(five*c2-one)*c2

        shl(10,l)= ((25.d0/12.d0)*(five*c3-two))*(five*c3-one)
     &                      *c3*(five*c2-one)*c2

        shl(11,l)= ((25.d0/24.d0)*(five*c3-three))
     &                     *(five*c3-two)*(five*c3-one)*c3*c2

        shl(12,l)= ((25.d0/24.d0)*(five*c3-three))
     &                      *(five*c3-two)*(five*c3-one)*c3*c1

        shl(13,l)= ((25.d0/12.d0)*(five*c3-two))
     &                      *(five*c3-one)*c3*(five*c1-one)*c1

        shl(14,l)= ((25.d0/12.d0)*(five*c3-one))
     &                       *c3*(five*c1-two)*(five*c1-one)*c1

        shl(15,l)= (25.d0/24.d0)*c3*(five*c1-three)*(five*c1-two)
     &         *(five*c1-one)*c1

        shl(16,l)=((125.d0/6.d0)*(five*c1-two))*(five*c1-one)*c1*c2*c3

        shl(17,l)=((125.d0/4.d0)*(five*c1-one))*(five*c2-one)*c1*c2*c3

        shl(18,l)=((125.d0/6.d0)*(five*c2-two))*(five*c2-one)*c1*c2*c3

        shl(19,l)= ((125.d0/4.d0)*(five*c2-one))*(five*c3-one)*c1*c2*c3

        shl(20,l)= ((125.d0/6.d0)*(five*c3-two))*(five*c3-one)*c1*c2*c3

        shl(21,l)= ((125.d0/4.d0)*(five*c1-one))*(five*c3-one)*c1*c2*c3

            end if

  200      continue
c

      return
      end

c-----------------------------------------------------------------------
      subroutine shapen(shl,nen,nenc)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension raone(8),xaone(8)
      dimension shlone(8,8),inod(8,8),inodc(8,8)
      dimension shl(64,64)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      if (nen.eq.1) xaone(1) = zero
      if (nenc.eq.1) raone(1) = zero
c
c
      if (nen.eq.4) then
         xaone(1) = -one
         xaone(2) =  one
         nenx=2
         neny=2
      endif
c
      if (nenc.eq.4) then
         raone(1) = -one
         raone(2) =  one
         nenrx=2
         nenry=2
      endif
c
c
      if(nen.eq.9) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
         nenx=3
         neny=3
      endif
c
      if(nenc.eq.9) then
         raone(1)= -one
         raone(2)= one
         raone(3)= zero
         nenrx=3
         nenry=3
      endif
c
c
      if (nen.eq.16) then
         xaone(1) = -one
         xaone(2) = one
         xaone(3) = -.333333333333333
         xaone(4) =  .333333333333333
         nenx=4
         neny=4
      endif
c
      if (nenc.eq.16) then
         raone(1) = -one
         raone(2) = one
         raone(3) = -.333333333333333
         raone(4) =  .333333333333333
         nenrx=4
         nenry=4
      endif
c
c
      if(nen.eq.25) then
         xaone(1)= -one
         xaone(2)=  one
         xaone(3)= -pt5
         xaone(4)= zero
         xaone(5)= pt5
         nenx=5
         neny=5
      endif
c
      if(nenc.eq.25) then
         raone(1)= -one
         raone(2)=  one
         raone(3)= -pt5
         raone(4)= zero
         raone(5)= pt5
         nenrx=5
         nenry=5
      endif
c
c
      if(nen.eq.36) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.600000000000000
         xaone(4) = -.200000000000000
         xaone(5) =  .200000000000000
         xaone(6) =  .600000000000000
         nenx=6
         neny=6
      endif
c
      if(nenc.eq.36) then
         raone(1) = -one
         raone(2) =  one
         raone(3) = -.600000000000000
         raone(4) = -.200000000000000
         raone(5) =  .200000000000000
         raone(6) =  .600000000000000
         nenrx=6
         nenry=6
      endif
c
c
      if(nen.eq.49) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.666666666666666
         xaone(4) = -.333333333333333
         xaone(5) = zero
         xaone(6) =  .333333333333333
         xaone(7) =  .666666666666666
         nenx=7
         neny=7
      endif
c
      if(nenc.eq.49) then
         raone(1) = -one
         raone(2) =  one
         raone(3) = -.666666666666666
         raone(4) = -.333333333333333
         raone(5) = zero
         raone(6) =  .333333333333333
         raone(7) =  .666666666666666
         nenrx=7
         nenry=7
      endif
c
c
      if(nen.eq.64) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -0.71428571428571
         xaone(4) = -0.42857142857143
         xaone(5) = -0.14285714285714
         xaone(6) = 0.14285714285714
         xaone(7) = 0.42857142857143
         xaone(8) = 0.71428571428571
         nenx=8
         neny=8
      endif
c
      if(nenc.eq.64) then
         raone(1) = -one
         raone(2) =  one
         raone(3) = -0.71428571428571
         raone(4) = -0.42857142857143
         raone(5) = -0.14285714285714
         raone(6) = 0.14285714285714
         raone(7) = 0.42857142857143
         raone(8) = 0.71428571428571
         nenrx=8
         nenry=8
      endif
c
c
      if(nen.eq.1) inod(1,1) = 1
      if(nen.eq.4) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(1,2) = 4
         inod(2,2) = 3
      end if
c
      if(nen.eq.9) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 7
c
         inod(1,3) = 8
         inod(2,3) = 6
         inod(3,3) = 9
      end if
c
c
c
      if(nen.eq.16) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 10
         inod(4,2) = 9
c
         inod(1,3) = 12
         inod(2,3) = 7
         inod(3,3) = 13
         inod(4,3) = 14
c
         inod(1,4) = 11
         inod(2,4) = 8
         inod(3,4) = 16
         inod(4,4) = 15
      end if
c
c
      if(nen.eq.25) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 13
         inod(4,2) = 12
         inod(5,2) = 11
c
         inod(1,3) = 16
         inod(2,3) = 8
         inod(3,3) = 17
         inod(4,3) = 18
         inod(5,3) = 19
c
         inod(1,4) = 15
         inod(2,4) = 9
         inod(3,4) = 24
         inod(4,4) = 25
         inod(5,4) = 20
c
         inod(1,5) = 14
         inod(2,5) = 10
         inod(3,5) = 23
         inod(4,5) = 22
         inod(5,5) = 21
      end if
c
c
      if(nen.eq.36) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 16
         inod(4,2) = 15
         inod(5,2) = 14
         inod(6,2) = 13
c
         inod(1,3) = 20
         inod(2,3) = 9
         inod(3,3) = 21
         inod(4,3) = 22
         inod(5,3) = 23
         inod(6,3) = 24
c
         inod(1,4) = 19
         inod(2,4) = 10
         inod(3,4) = 32
         inod(4,4) = 33
         inod(5,4) = 34
         inod(6,4) = 25
c
         inod(1,5) = 18
         inod(2,5) = 11
         inod(3,5) = 31
         inod(4,5) = 36
         inod(5,5) = 35
         inod(6,5) = 26
c
         inod(1,6) = 17
         inod(2,6) = 12
         inod(3,6) = 30
         inod(4,6) = 29
         inod(5,6) = 28
         inod(6,6) = 27
c
      end if
c
c
      if(nen.eq.49) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
         inod(7,1) = 9
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 19
         inod(4,2) = 18
         inod(5,2) = 17
         inod(6,2) = 16
         inod(7,2) = 15
c
         inod(1,3) = 24
         inod(2,3) = 10
         inod(3,3) = 25
         inod(4,3) = 26
         inod(5,3) = 27
         inod(6,3) = 28
         inod(7,3) = 29
c
         inod(1,4) = 23
         inod(2,4) = 11
         inod(3,4) = 40
         inod(4,4) = 41
         inod(5,4) = 42
         inod(6,4) = 43
         inod(7,4) = 30
c
         inod(1,5) = 22
         inod(2,5) = 12
         inod(3,5) = 39
         inod(4,5) = 48
         inod(5,5) = 49
         inod(6,5) = 44
         inod(7,5) = 31
c
         inod(1,6) = 21
         inod(2,6) = 13
         inod(3,6) = 38
         inod(4,6) = 47
         inod(5,6) = 46
         inod(6,6) = 45
         inod(7,6) = 32
c
         inod(1,7) = 20
         inod(2,7) = 14
         inod(3,7) = 37
         inod(4,7) = 36
         inod(5,7) = 35
         inod(6,7) = 34
         inod(7,7) = 33
c
      end if
c
c
      if(nen.eq.64) then
         inod(1,1) = 1
         inod(2,1) = 2
         inod(3,1) = 5
         inod(4,1) = 6
         inod(5,1) = 7
         inod(6,1) = 8
         inod(7,1) = 9
         inod(8,1) = 10
c
         inod(1,2) = 4
         inod(2,2) = 3
         inod(3,2) = 22
         inod(4,2) = 21
         inod(5,2) = 20
         inod(6,2) = 19
         inod(7,2) = 18
         inod(8,2) = 17
c
         inod(1,3) = 28
         inod(2,3) = 11
         inod(3,3) = 29
         inod(4,3) = 30
         inod(5,3) = 31
         inod(6,3) = 32
         inod(7,3) = 33
         inod(8,3) = 34
c
         inod(1,4) = 27
         inod(2,4) = 12
         inod(3,4) = 48
         inod(4,4) = 49
         inod(5,4) = 50
         inod(6,4) = 51
         inod(7,4) = 52
         inod(8,4) = 35
c
         inod(1,5) = 26
         inod(2,5) = 13
         inod(3,5) = 47
         inod(4,5) = 60
         inod(5,5) = 61
         inod(6,5) = 62
         inod(7,5) = 53
         inod(8,5) = 36
c
         inod(1,6) = 25
         inod(2,6) = 14
         inod(3,6) = 46
         inod(4,6) = 59
         inod(5,6) = 64
         inod(6,6) = 63
         inod(7,6) = 54
         inod(8,6) = 37
c
         inod(1,7) = 24
         inod(2,7) = 15
         inod(3,7) = 45
         inod(4,7) = 58
         inod(5,7) = 57
         inod(6,7) = 56
         inod(7,7) = 55
         inod(8,7) = 38
c
         inod(1,8) = 23
         inod(2,8) = 16
         inod(3,8) = 44
         inod(4,8) = 43
         inod(5,8) = 42
         inod(6,8) = 41
         inod(7,8) = 40
         inod(8,8) = 39
c
      end if
c
c
c
      if(nenc.eq.1) inodc(1,1) = 1
      if(nenc.eq.4) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(1,2) = 4
         inodc(2,2) = 3
      end if
c
      if(nenc.eq.9) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(3,1) = 5
c
         inodc(1,2) = 4
         inodc(2,2) = 3
         inodc(3,2) = 7
c
         inodc(1,3) = 8
         inodc(2,3) = 6
         inodc(3,3) = 9
      end if
c
c
c
      if(nenc.eq.16) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(3,1) = 5
         inodc(4,1) = 6
c
         inodc(1,2) = 4
         inodc(2,2) = 3
         inodc(3,2) = 10
         inodc(4,2) = 9
c
         inodc(1,3) = 12
         inodc(2,3) = 7
         inodc(3,3) = 13
         inodc(4,3) = 14
c
         inodc(1,4) = 11
         inodc(2,4) = 8
         inodc(3,4) = 16
         inodc(4,4) = 15
      end if
c
c
      if(nenc.eq.25) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(3,1) = 5
         inodc(4,1) = 6
         inodc(5,1) = 7
c
         inodc(1,2) = 4
         inodc(2,2) = 3
         inodc(3,2) = 13
         inodc(4,2) = 12
         inodc(5,2) = 11
c
         inodc(1,3) = 16
         inodc(2,3) = 8
         inodc(3,3) = 17
         inodc(4,3) = 18
         inodc(5,3) = 19
c
         inodc(1,4) = 15
         inodc(2,4) = 9
         inodc(3,4) = 24
         inodc(4,4) = 25
         inodc(5,4) = 20
c
         inodc(1,5) = 14
         inodc(2,5) = 10
         inodc(3,5) = 23
         inodc(4,5) = 22
         inodc(5,5) = 21
      end if
c
c
      if(nenc.eq.36) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(3,1) = 5
         inodc(4,1) = 6
         inodc(5,1) = 7
         inodc(6,1) = 8
c
         inodc(1,2) = 4
         inodc(2,2) = 3
         inodc(3,2) = 16
         inodc(4,2) = 15
         inodc(5,2) = 14
         inodc(6,2) = 13
c
         inodc(1,3) = 20
         inodc(2,3) = 9
         inodc(3,3) = 21
         inodc(4,3) = 22
         inodc(5,3) = 23
         inodc(6,3) = 24
c
         inodc(1,4) = 19
         inodc(2,4) = 10
         inodc(3,4) = 32
         inodc(4,4) = 33
         inodc(5,4) = 34
         inodc(6,4) = 25
c
         inodc(1,5) = 18
         inodc(2,5) = 11
         inodc(3,5) = 31
         inodc(4,5) = 36
         inodc(5,5) = 35
         inodc(6,5) = 26
c
         inodc(1,6) = 17
         inodc(2,6) = 12
         inodc(3,6) = 30
         inodc(4,6) = 29
         inodc(5,6) = 28
         inodc(6,6) = 27
c
      end if
c
c
      if(nenc.eq.49) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(3,1) = 5
         inodc(4,1) = 6
         inodc(5,1) = 7
         inodc(6,1) = 8
         inodc(7,1) = 9
c
         inodc(1,2) = 4
         inodc(2,2) = 3
         inodc(3,2) = 19
         inodc(4,2) = 18
         inodc(5,2) = 17
         inodc(6,2) = 16
         inodc(7,2) = 15
c
         inodc(1,3) = 24
         inodc(2,3) = 10
         inodc(3,3) = 25
         inodc(4,3) = 26
         inodc(5,3) = 27
         inodc(6,3) = 28
         inodc(7,3) = 29
c
         inodc(1,4) = 23
         inodc(2,4) = 11
         inodc(3,4) = 40
         inodc(4,4) = 41
         inodc(5,4) = 42
         inodc(6,4) = 43
         inodc(7,4) = 30
c
         inodc(1,5) = 22
         inodc(2,5) = 12
         inodc(3,5) = 39
         inodc(4,5) = 48
         inodc(5,5) = 49
         inodc(6,5) = 44
         inodc(7,5) = 31
c
         inodc(1,6) = 21
         inodc(2,6) = 13
         inodc(3,6) = 38
         inodc(4,6) = 47
         inodc(5,6) = 46
         inodc(6,6) = 45
         inodc(7,6) = 32
c
         inodc(1,7) = 20
         inodc(2,7) = 14
         inodc(3,7) = 37
         inodc(4,7) = 36
         inodc(5,7) = 35
         inodc(6,7) = 34
         inodc(7,7) = 33
c
      end if
c
c
      if(nenc.eq.64) then
         inodc(1,1) = 1
         inodc(2,1) = 2
         inodc(3,1) = 5
         inodc(4,1) = 6
         inodc(5,1) = 7
         inodc(6,1) = 8
         inodc(7,1) = 9
         inodc(8,1) = 10
c
         inodc(1,2) = 4
         inodc(2,2) = 3
         inodc(3,2) = 22
         inodc(4,2) = 21
         inodc(5,2) = 20
         inodc(6,2) = 19
         inodc(7,2) = 18
         inodc(8,2) = 17
c
         inodc(1,3) = 28
         inodc(2,3) = 11
         inodc(3,3) = 29
         inodc(4,3) = 30
         inodc(5,3) = 31
         inodc(6,3) = 32
         inodc(7,3) = 33
         inodc(8,3) = 34
c
         inodc(1,4) = 27
         inodc(2,4) = 12
         inodc(3,4) = 48
         inodc(4,4) = 49
         inodc(5,4) = 50
         inodc(6,4) = 51
         inodc(7,4) = 52
         inodc(8,4) = 35
c
         inodc(1,5) = 26
         inodc(2,5) = 13
         inodc(3,5) = 47
         inodc(4,5) = 60
         inodc(5,5) = 61
         inodc(6,5) = 62
         inodc(7,5) = 53
         inodc(8,5) = 36
c
         inodc(1,6) = 25
         inodc(2,6) = 14
         inodc(3,6) = 46
         inodc(4,6) = 59
         inodc(5,6) = 64
         inodc(6,6) = 63
         inodc(7,6) = 54
         inodc(8,6) = 37
c
         inodc(1,7) = 24
         inodc(2,7) = 15
         inodc(3,7) = 45
         inodc(4,7) = 58
         inodc(5,7) = 57
         inodc(6,7) = 56
         inodc(7,7) = 55
         inodc(8,7) = 38
c
         inodc(1,8) = 23
         inodc(2,8) = 16
         inodc(3,8) = 44
         inodc(4,8) = 43
         inodc(5,8) = 42
         inodc(6,8) = 41
         inodc(7,8) = 40
         inodc(8,8) = 39
c
      end if
c
      do 100 l = 1, nenrx
         r = raone(l)
c
         if(nenx.eq.1) then
            shlone(1,l) = zero
            go to 100
         endif
c
         do 50 i = 1, nenx
            aa = one
            bb = one
            aax = zero
            do 40 j =1, nenx
               daj = one
               if (i .ne. j)then
                  aa = aa * ( r - xaone(j))
                  bb = bb * ( xaone(i) - xaone(j))
               endif
 40         continue
            shlone(i,l) = aa/bb
 50      continue
c
 100  continue
c
      do ly=1,nenrx
         do lx=1,nenry
            l = inodc(lx,ly)
            do iy=1,neny
               do ix=1,nenx
                  j = inod(ix,iy)
                  shl(j,l) = shlone(ix,lx)*shlone(iy,ly)
               end do
            end do
         end do
      end do
c
      return
      end


c-----------------------------------------------------------------------
      subroutine shapesd(shl,nen,nenc)
c-----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
c
      dimension raone(8),xaone(8)
      dimension shl(8,8)
      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c
      if (nen.eq.1) xaone(1) = zero
      if (nenc.eq.1) raone(1) = zero
c
      if (nen.eq.2) then
         xaone(1) = -one
         xaone(2) =  one
      endif
c
      if (nenc.eq.2) then
         raone(1) = -one
         raone(2) =  one
      endif
c
      if(nen.eq.3) then
         xaone(1)= -one
         xaone(2)= one
         xaone(3)= zero
      endif
c
      if(nenc.eq.3) then
         raone(1)= -one
         raone(2)= one
         raone(3)= zero
      endif
c
      if (nen.eq.4) then
         xaone(1) = -one
         xaone(2) = one
         xaone(3) = -.333333333333333
         xaone(4) =  .333333333333333
      endif
c
      if (nenc.eq.4) then
         raone(1) = -one
         raone(2) = one
         raone(3) = -.333333333333333
         raone(4) =  .333333333333333
      endif
c
c
      if(nen.eq.5) then
         xaone(1)= -one
         xaone(2)=  one
         xaone(3)= -pt5
         xaone(4)= zero
         xaone(5)= pt5
      endif
c
      if(nenc.eq.5) then
         raone(1)= -one
         raone(2)=  one
         raone(3)= -pt5
         raone(4)= zero
         raone(5)= pt5
      endif
c
c
      if(nen.eq.6) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.600000000000000
         xaone(4) = -.200000000000000
         xaone(5) =  .200000000000000
         xaone(6) =  .600000000000000
      endif
c
      if(nenc.eq.6) then
         raone(1) = -one
         raone(2) =  one
         raone(3) = -.600000000000000
         raone(4) = -.200000000000000
         raone(5) =  .200000000000000
         raone(6) =  .600000000000000
      endif
c
      if(nen.eq.7) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -.666666666666666
         xaone(4) = -.333333333333333
         xaone(5) = zero
         xaone(6) =  .333333333333333
         xaone(7) =  .666666666666666
      endif
c
      if(nenc.eq.7) then
         raone(1) = -one
         raone(2) =  one
         raone(3) = -.666666666666666
         raone(4) = -.333333333333333
         raone(5) = zero
         raone(6) =  .333333333333333
         raone(7) =  .666666666666666
      endif
c
      if(nen.eq.8) then
         xaone(1) = -one
         xaone(2) =  one
         xaone(3) = -0.71428571428571
         xaone(4) = -0.42857142857143
         xaone(5) = -0.14285714285714
         xaone(6) = 0.14285714285714
         xaone(7) = 0.42857142857143
         xaone(8) = 0.71428571428571
      endif
c
      if(nenc.eq.8) then
         raone(1) = -one
         raone(2) =  one
         raone(3) = -0.71428571428571
         raone(4) = -0.42857142857143
         raone(5) = -0.14285714285714
         raone(6) = 0.14285714285714
         raone(7) = 0.42857142857143
         raone(8) = 0.71428571428571
      endif
c
c     loop on nodes to compute shape functions
c
      do 100 l = 1, nenc
         r = raone(l)
c
         if(nen.eq.1) then
            shl(1,l) = zero
            go to 100
         endif
c
         do i = 1, nen
            aa = one
            bb = one
            aax = zero
            do j =1, nen
               daj = one
               if (i .ne. j)then
                  aa = aa * ( r - xaone(j))
                  bb = bb * ( xaone(i) - xaone(j))
               endif
            end do
            shl(i,l) = aa/bb
         end do
c
 100  continue
c
      return
      end

c-----------------------------------------------------------------------
c$$$      subroutine gnusol(ien   ,x     ,xl   ,
c$$$     &                  d     ,dl    ,mat   ,
c$$$     &                  c     ,ipar  ,dlf   ,
c$$$     &                  dlp   ,dsfl  ,det   ,
c$$$     &                  shl   , shg  ,wt    ,
c$$$     &                  detc  ,shlc  , shgc ,
c$$$     &                  ddis  ,detp  ,shlp  ,
c$$$     &                  shgp ,
c$$$c
c$$$     &                  shln  ,shgn  ,
c$$$     &                  detn  ,shlb  ,shgb  ,
c$$$     &                  detpn ,shlpn ,shgpn ,
c$$$     &                  idside,xls   ,idlsd ,
c$$$     &                  grav  ,wn    ,
c$$$c
c$$$     &                  numel ,neesq ,nen   ,nsd   ,
c$$$     &                  nesd  ,nint  ,neg   ,nrowsh,
c$$$     &                  ned   ,nee   ,numnp ,ndof  ,
c$$$     &                  ncon  ,nencon,necon ,index ,
c$$$     &                  nints,iwrite ,iplt ,nenp   ,
c$$$     &                  nside,nnods ,nenlad,npars  ,
c$$$     &                  nmultp, nodsp)
c$$$c-----------------------------------------------------------------------
c$$$      implicit real*8 (a-h,o-z)
c$$$c
c$$$      character *1 tab
c$$$      dimension ien(nen,*),x(nsd,*),xl(nesd,*),d(ndof,*),dl(ned,*),
c$$$     &     mat(*),c(10,*),ipar(nodsp,*),dlf(ncon,*) ,
c$$$     &     dsfl(ncon,nencon,*),ddis(ned,nenp,*)
c$$$      dimension u(6),dux(6),duy(6),ue(6),duex(6),duey(6)
c$$$      dimension shl(3,nen,*),shg(3,nen,*),det(*),wt(*)
c$$$      dimension shlc(3,nencon,*),shgc(3,nencon,*),detc(*)
c$$$      dimension el2(6),epri(6),eprix(6),epriy(6),el2el(6),
c$$$     &     eprxel(6),epryel(6)
c$$$      dimension detp(*),shlp(3,nenp,*),shgp(3,nenp,*),dlp(ned,*)
c$$$c
c$$$      dimension detn(*),shlb(3,nenlad,*),shgb(3,nenlad,*),
c$$$     &     detpn(*),shlpn(3,npars,*),shgpn(3,npars,*),
c$$$     &     idside(nside,*),xls(nesd,*),idlsd(*)
c$$$      dimension dls(12),grav(*),wn(*)
c$$$      dimension shln(3,nnods,*),shgn(3,nnods,*)
c$$$c
c$$$      common /consts/ zero,pt1667,pt25,pt5,one,two,three,four,five,six
c$$$      common /iounit/ iin,iout,iecho,ioupp,itest1,itest2
c$$$c
c$$$      pi=4.*datan(1.d00)
c$$$      dpi=2.d00*pi
c$$$c
c$$$      gf1=grav(1)
c$$$      gf2=grav(2)
c$$$      gf3=grav(3)
c$$$      eps=c(9,1)
c$$$      tab=char(9)
c$$$      xint=0.d00
c$$$      yint=0.d00
c$$$c
c$$$      call clear ( el2, ncon )
c$$$      call clear (eprix, ncon )
c$$$      call clear (epriy, ncon )
c$$$c
c$$$      ipred=17
c$$$      open(unit=ipred, file= 'predesc.dat')
c$$$c
c$$$c     loop on elements
c$$$c
c$$$      do n=1,numel
c$$$c
c$$$         call local(ien(1,n),x,xl,nen,nsd,nesd)
c$$$         call local(ipar(1,n),d,dl,nodsp,ndof,ned)
c$$$c
c$$$         do i=1,nenp
c$$$            dlp(1,i) = ddis(1,i,n)
c$$$            write(ipred,991) xl(1,i),xl(2,i),dlp(1,i)
c$$$         end do
c$$$         write(ipred,991) xl(1,1),xl(2,1),dlp(1,1)
c$$$         write(ipred,*)
c$$$         write(ipred,*)
c$$$ 991     format(6e15.5)
c$$$c
c$$$      end do
c$$$c
c$$$      return
c$$$      end

c-----------------------------------------------------------------------
      subroutine dumpmsh(ien, x, xl, idside, lado, numel, numnp,
     &                   nen, nsd, nesd, nedge, npars, nside)
c-----------------------------------------------------------------------
c     program to dump mesh to file and screen
c-----------------------------------------------------------------------
      dimension ien(nen,*),x(nsd,*),xl(nesd,*)
      dimension lado(nside,*)
c      integer elfaces(nedge,npars)
      integer iboolf(nedge)
c
      do i=1,nedge
         iboolf(i)=0
      end do
c
      open(123, file='malha.msh')
      write(*,*) "dump mesh to screen/file",nsd,nesd
      write(123,'(a)') "$MeshFormat"
      write(123,'(a)') "2.2 0 8"
      write(123,'(a)') "$EndMeshFormat"
c
      write(*,*) " coordenadas", nen, nsd, nesd
      write(123,'(a)') "$Nodes$"
      do np=1,numnp
         write(*,456) np, (x(i,np), i=1,nsd)
      end do
      write(123,*) numnp
      do np=1,numnp
         write(123,456) np, (x(i,np), i=1,nsd)
      end do
 456  format(' ',I6,30F10.6)
      write(123,'(a)') "$EndNodes"
c
      write(123,'(a)') "$Elements"
      write(123,*) numel
c      write(123,*) nedge+numel
      write(*,*) " conectividade - elementos"
      do nel=1,numel
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
         write(*,*) nel, (ien(j, nel), j=1,nen)
      end do
c      write(*,*) " faces por elemento"
c      do nel=1,numel
c         call local(ien(1,nel),x,xl,nen,nsd,nesd)
c         write(*,*) nel, (lado(j,nel), j=1,nside)
c      end do
c      write(*,*) " numeracao local das faces"
c      do il=1,nside
c         write(*,*) il, (idside(il,j),j=1,4)
c      end do
c      write(*,*)
c
c      write(*,*) " conectividade - faces"
c      kel = 1
c      do nel=1,numel
c         call local(ien(1,nel),x,xl,nen,nsd,nesd)
c         do ns=1,nside
c            ns1=idside(ns,1)
c            ns2=idside(ns,2)
c            ns3=idside(ns,3)
c            ns4=idside(ns,4)
c
c            nl1=ien(ns1,nel)
c            nl2=ien(ns2,nel)
c            nl3=ien(ns3,nel)
c            nl4=ien(ns4,nel)
c
c            iface=lado(ns,nel)
c            write(*,*) nel, ns, iface, nl1, nl2, nl3, nl4
c            if(iboolf(iface).eq.0) then
c               iboolf(iface) = 1
c               elfaces(iface,1) = nl1
c               elfaces(iface,2) = nl2
c               elfaces(iface,3) = nl3
c               elfaces(iface,4) = nl4
c            end if
c            kel = kel + 1
c         end do
c      end do

c      do ifc=1,nedge
c         nl1=elfaces(ifc,1)
c         nl2=elfaces(ifc,2)
c         nl3=elfaces(ifc,3)
c         nl4=elfaces(ifc,4)
c         write(123,789) ifc, 3, 2, 100, 200, nl1, nl2, nl3, nl4
c      end do
c     kel = nedge + 1
      kel = 1
      do nel=1,numel
         call local(ien(1,nel),x,xl,nen,nsd,nesd)
         write(123,789) nel,5,2,100,200,(ien(j,nel),j=1,nen)
      end do
      write(123,'(a)') "$EndElements"
      close(123)
 789  format(' ', 16I5)
c
      close(123)
c
      return
      end
