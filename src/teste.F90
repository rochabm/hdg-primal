c
      program testdmplex
c      
#include <petsc/finclude/petsc.h>
#include <petsc/finclude/petscdmplex.h>
      use petsc
      use petscdmplex
c      
      implicit real*8 (a-h,o-z)
c      

c      
      DM dm
      PetscErrorCode ierr
      PetscInt hbcells(3)
      PetscInt, target, dimension(6) :: cone
      PetscInt, pointer :: ptr(:)
      PetscMPIInt size,rank
      PetscReal, dimension(3) :: low,upp

      PetscReal xnodes(12)
      PetscInt ielement(16)
c
c     definitions
c
      ndim  = 3  ! number of dimensions 
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
c     DMPlex
c
      hbcells(1)=2
      hbcells(2)=1
      hbcells(3)=1
c
      low(1)=0.0
      low(2)=0.0
      low(3)=0.0
c      
      upp(1)=1.0
      upp(2)=1.0
      upp(3)=1.0
c     
      call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE,
     &      hbcells,low,upp,DM_BOUNDARY_NONE, PETSC_TRUE, dm, ierr)
c     
c      call DMPlexCreateHexBoxMesh(PETSC_COMM_WORLD, ndim, hbcells,
c     &   DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm, ierr)      
c     CHKERRQ(ierr)

c$$$      do i=1,12
c$$$         xnodes(i) = 0.0
c$$$      end do
c$$$
c$$$      ielement(1) = 0
c$$$      ielement(2) = 1
c$$$      ielement(3) = 4
c$$$      ielement(4) = 3
c$$$      ielement(5) = 6
c$$$      ielement(6) = 7
c$$$      ielement(7) = 10
c$$$      ielement(8) = 9
c$$$c
c$$$      ielement( 9) = 1
c$$$      ielement(10) = 2
c$$$      ielement(11) = 5
c$$$      ielement(12) = 4
c$$$      ielement(13) = 7
c$$$      ielement(14) = 8
c$$$      ielement(15) = 11
c$$$      ielement(16) = 10
c$$$      ndim=3
c$$$      numel=2
c$$$      numnp=12
c$$$      nen=8
c$$$      call DMPlexCreateFromCellList(PETSC_COMM_WORLD,
c$$$     &                              ndim,numel,numnp,nen,PETSC_TRUE,
c$$$     &                              ielement,ndim,xnodes,dm,ierr)
c$$$      CHKERRQ(ierr)      

      call DMView(dm,PETSC_VIEWER_STDOUT_WORLD,ierr)
      CHKERRQ(ierr)
      stop
c
c     trying to get the closure of cell 0
c
      ptr => cone

      write(*,*) "Check GetCone"
      do ie=0,7
         call DMPlexGetCone(dm, ie , ptr, ierr)
         CHKERRQ(ierr)
         write(*,*) 'cell',ie,ptr
         call DMPlexRestoreCone(dm, ie , ptr, ierr)
         CHKERRQ(ierr)         
      end do            
c
c     end
c      
      call PetscFinalize(ierr)
      CHKERRQ(ierr)
      
      end program testdmplex 
