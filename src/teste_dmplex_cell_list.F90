c ----------------------------------------------------------------------
c     This program is used to test the DMPlex implementation in PETSc
c     The plex is created using CreateFromCellList
c     Bernardo M. Rocha, April, 2018
c ----------------------------------------------------------------------
      program testdmplex
c     
#include <petsc/finclude/petsc.h>
#include <petsc/finclude/petscdmplex.h>
      use petscdmplex
      use petsc
c
      implicit real*8 (a-h,o-z)
c      
      DM dm
      PetscErrorCode ierr
      PetscInt icell
      PetscInt ipstart, ipend
      PetscInt hbcells(3)
      PetscMPIInt size,rank
      PetscInt pStart, pEnd
      PetscInt fStart, fEnd, f
      PetscInt cStart, cEnd, c
      PetscInt vStart, vEnd, v
      PetscInt eStart, eEnd, e
      PetscInt npts, nsec, idof, ioff
      PetscInt, pointer :: pp(:)
      PetscInt, target, dimension(54) :: pts
c
c     simple 2x2x2 case hardcoded
c     
      PetscInt ielem(8*8)  !nen x numel
      PetscInt ielem2(8*8) !nen x numel      
      PetscScalar xnodes(3*27) 
      PetscSection sec
c
      real*8 xx(3),yy(3),zz(3)
      real*8 xcoords(27,3)
c
c     definitions
c
      ndim  = 3  ! number of dimensions 
      nen   = 8  ! number of element nodes
      numel = 8  ! number of elements (2x2x2)
      numnp = 27 ! number of nodes (3x3x3) 
c     
c     coordinates
c
      xdx = 0.5d0
      do i=1,3
         xx(i) = (i-1)*xdx
         yy(i) = (i-1)*xdx
         zz(i) = (i-1)*xdx
      end do
c
      id = 1
      do k=1,3
         z = zz(k)
         do j=1,3
            y = yy(j) 
            do i=1,3               
               x = xx(i)
               xcoords(id,1) = x
               xcoords(id,2) = y
               xcoords(id,3) = z
               id = id + 1
            end do
         end do
      end do
c
      id = 0
      do i=1,numnp
         id = id + 1
         xnodes(id) = xcoords(i,1)
         id = id + 1
         xnodes(id) = xcoords(i,2)
         id = id + 1
         xnodes(id) = xcoords(i,3)
      end do
c
      write(*,*) "coordinates"
      do i=1,3*numnp,3
         write(*,*) xnodes(i), xnodes(i+1), xnodes(i+2)
      end do   
c
c     element connectivity
c    
c        7------6
c       /|       /|
c      / |      / |
c     4------5  |
c     |   |     | |
c     |   |     | |
c     |  1------2
c     |  /      | /
c     | /       |/
c     0------3
c
c     HARDCODED (swapped local nodes 2 and 4)
c      
      ielem(1) = 1;  ielem(4) = 2;  ielem(3) = 5;  ielem(2) = 4
      ielem(5) = 10; ielem(6) = 11; ielem(7) = 14; ielem(8) = 13
c
      ielem(9)  = 2;  ielem(12) = 3;  ielem(11) = 6;  ielem(10) = 5
      ielem(13) = 11; ielem(14) = 12; ielem(15) = 15; ielem(16) = 14
c
      ielem(17) = 4;  ielem(20) = 5;  ielem(19) = 8;  ielem(18) = 7
      ielem(21) = 13; ielem(22) = 14; ielem(23) = 17; ielem(24) = 16
c
      ielem(25) = 5;  ielem(28) = 6;  ielem(27) = 9;  ielem(26) = 8
      ielem(29) = 14; ielem(30) = 15; ielem(31) = 18; ielem(32) = 17
c
      ielem(33) = 10; ielem(36) = 11; ielem(35) = 14; ielem(34) = 13
      ielem(37) = 19; ielem(38) = 20; ielem(39) = 23; ielem(40) = 22
c
      ielem(41) = 11; ielem(44) = 12; ielem(43) = 15; ielem(42) = 14
      ielem(45) = 20; ielem(46) = 21; ielem(47) = 24; ielem(48) = 23
c
      ielem(49) = 13; ielem(52) = 14; ielem(51) = 17; ielem(50) = 16
      ielem(53) = 22; ielem(54) = 23; ielem(55) = 26; ielem(56) = 25
c
      ielem(57) = 14; ielem(60) = 15; ielem(59) = 18; ielem(58) = 17;
      ielem(61) = 23; ielem(62) = 24; ielem(63) = 27; ielem(64) = 26
c
      do i=1,64
         ielem(i) = ielem(i) - 1
      end do
c
      write(*,*) "elements"
      do i=1,8
         k = (i-1)*8
         write(*,*) ielem(k+1),ielem(k+2),ielem(k+3),ielem(k+4),
     &              ielem(k+5),ielem(k+6),ielem(k+7),ielem(k+8)
      end do
c
c     testing DMPlex in Fortran
c      
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
         call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
         if (rank .eq. 0) then
            write(6,*) 'This is a uniprocessor example only!'
         endif
      endif
c
c     test DMPlex
c      
      call DMPlexCreateFromCellList(PETSC_COMM_WORLD,
     &                              ndim,numel,numnp,nen,PETSC_TRUE,
     &                              ielem,ndim,xnodes,dm,ierr)
      CHKERRQ(ierr)

      call DMView(dm,PETSC_VIEWER_STDOUT_WORLD,ierr)
      CHKERRQ(ierr)          
      
      call DMPlexGetChart(dm,ipstart,ipend,ierr)
      CHKERRQ(ierr)
      
      write(*,*) "DMPlex chart start=",ipstart
      write(*,*) "DMPlex chart end=",ipend
c
c     trying to get the cone of element 0
c
      icsz = 0
      icell = 0      
      call DMPlexGetConeSize(dm, icell, icsz, ierr)
      CHKERRQ(ierr)

      write(*,*) "cone for cell", icell, " has size",icsz
      
      !pts => clos
      pp => pts
      np = 0
      
      call DMPlexGetTransitiveClosure(dm,0,PETSC_TRUE,pp,ierr)
      
c      write(*,*) " closure ", clos(1)
c      do i=1, 10, 2
c      c   write(*,*) clos(i)
c      cend do
    
      call DMPlexRestoreTransitiveClosure(dm,0,PETSC_TRUE,pp,ierr)
c
c     end
c      
      call PetscFinalize(ierr)
      CHKERRQ(ierr)
      
      end program testdmplex 



c$$$c
c$$$c     test F90 interface for Vector
c$$$c      
c$$$      write(*,*) "testando vetor"
c$$$      x = 1.23
c$$$      call VecCreate(PETSC_COMM_WORLD,v,ierr)
c$$$      call VecSetType(v,VECSEQ,ierr)
c$$$      call VecSetSizes(v,3,PETSC_DECIDE,ierr)
c$$$      call VecSetValue(v,0,x,INSERT_VALUES,ierr)
c$$$      call VecSetValue(v,1,x,INSERT_VALUES,ierr)
c$$$      call VecSetValue(v,2,x,INSERT_VALUES,ierr)
c$$$      call VecAssemblyBegin(v,ierr)
c$$$      call VecAssemblyEnd(v,ierr)      
c$$$      call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr)
c$$$      
c$$$      call VecGetArrayF90(v,xx_v,ierr)
c$$$      xx_v(1) = 0.0
c$$$      xx_v(2) = 0.0
c$$$      xx_v(3) = 0.0
c$$$      call VecRestoreArrayF90(v,xx_v,ierr)
c$$$      call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr)
c$$$      write(*,*)
c$$$      write(*,*)
c$$$
c$$$      call VecDestroy(v,ierr)      
