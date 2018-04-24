c ----------------------------------------------------------------------
c     This program is used to test the DMPlex implementation in PETSc
c     The plex is created using the CreateBoxMesh
c     Bernardo M. Rocha, April, 2018
c ----------------------------------------------------------------------
      program testdmplex
c      
#include <petsc/finclude/petsc.h>
#include <petsc/finclude/petscdmplex.h>
      use petsc
      use petscdmplex
c      
      implicit none
c
      integer i, j, ie, ic, ix, ndim, npts, nsec, idof, ioff
c      
      DM dm
      PetscErrorCode ierr
      PetscInt hbcells(3)
      PetscInt, target, dimension(6) :: cone
      PetscInt, pointer :: ptr(:)
      PetscMPIInt size,rank
      PetscReal, dimension(3) :: low, upp
      PetscReal xnodes(12)
      PetscInt ielement(16)
      PetscInt pStart, pEnd
      PetscInt fStart, fEnd, f
      PetscInt cStart, cEnd, c
      PetscInt vStart, vEnd, v
      PetscInt eStart, eEnd, e
      PetscInt, pointer :: pp(:)
      PetscInt, target, dimension(54) :: pts
      PetscSection sec
c
c     definitions
c
      ndim  = 3 ! dimension
      npts = 27 ! internal for pts,pp
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
      hbcells(1) = 2
      hbcells(2) = 2
      hbcells(3) = 2
c
      low(1) = 0.0
      low(2) = 0.0
      low(3) = 0.0
c      
      upp(1)=1.0
      upp(2)=1.0
      upp(3)=1.0
c     
      call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE,
     &     hbcells,low,upp,DM_BOUNDARY_NONE, PETSC_TRUE, dm, ierr)
      CHKERRQ(ierr)      
c    
      call DMView(dm,PETSC_VIEWER_STDOUT_WORLD,ierr)
      CHKERRQ(ierr)
c
c     extract all the relevant information from nodes, cells, faces and edges
c     
      call DMPlexGetChart(dm, pStart, pEnd, ierr)
      CHKERRQ(ierr)
      call DMPlexGetHeightStratum(dm, 0, cStart, cEnd, ierr) !heigt=0 cells
      CHKERRQ(ierr)
      call DMPlexGetDepthStratum(dm, 2, fStart, fEnd, ierr) !depth=2 faces
      CHKERRQ(ierr)
      call DMPlexGetDepthStratum(dm, 1, eStart, eEnd, ierr) !depth=1 edges
      CHKERRQ(ierr)      
      call DMPlexGetDepthStratum(dm, 0, vStart, vEnd, ierr) !depth=0 vertices
      CHKERRQ(ierr)
c
      write(*,*)
      write(*,*) "Cells", cStart, cEnd
      write(*,*) "Faces", fStart, fEnd
      write(*,*) "Edges", eStart, eEnd
      write(*,*) "Verts", vStart, vEnd
      write(*,*)     
c
c     trying to get the closure of cell 0
c
      ptr => cone

      write(*,*) "GetCone"
      do ie=0,7
         call DMPlexGetCone(dm, ie , ptr, ierr)
         CHKERRQ(ierr)
         write(*,*) 'cell',ie,ptr
         call DMPlexRestoreCone(dm, ie , ptr, ierr)
         CHKERRQ(ierr)         
      end do
c
c     transitive closure
c      
      do ic=cStart, cEnd-1
         pp => pts
         
         call DMPlexGetTransitiveClosure(dm,ic,PETSC_TRUE,pp,ierr)
         CHKERRQ(ierr)

         write(*,*)
         write(*,'(A,2I5)') " Cell", ic
                  
         write(*,'(A)', advance="no") " Faces"
         do j=1,2*npts,2
            ix = pp(j)
            if(fStart.le.ix .and. ix.lt.fEnd) then
               write(*,'(2I5)', advance="no") ix
            end if
         end do
         write(*,*)

         write(*,'(A)', advance="no") " Edges"
         do j=1,2*npts,2
            ix = pp(j)
            if(eStart.le.ix .and. ix.lt.eEnd) then
               write(*,'(I5)', advance="no") ix
            end if
         end do
         write(*,*)

         write(*,'(A)', advance="no") " Verts"
         do j=1,2*npts,2
            ix = pp(j)
            if(vStart.le.ix .and. ix.lt.vEnd) then
               write(*,'(I5)', advance="no") ix
            end if
         end do
         write(*,*)

         call DMPlexRestoreTransitiveClosure(dm,ic,PETSC_TRUE,pp,ierr)
         CHKERRQ(ierr)
      end do
c
c     DOF numbering
c      
      write(*,*) ""
      write(*,*) "DOF numbering"
      write(*,*) ""
c      
      call PetscSectionCreate(PETSC_COMM_WORLD,sec,ierr)
      CHKERRQ(ierr)
      call PetscSectionSetChart(sec,pStart,pEnd,ierr)
      CHKERRQ(ierr)
      
      write(*,*)
      write(*,'(A,I10,A,I10,A)') "Section chart [",pStart,",",pEnd,")"      
      write(*,*)      
c     
c     fixed for Q2
c     (dof: 1 per node, 1 per edge and 1 per face)
c
      do v=vStart,vEnd-1
         call PetscSectionSetDof(sec,v,1,ierr)
         CHKERRQ(ierr)
      end do
      do e=eStart,eEnd-1
         call PetscSectionSetDof(sec,e,1,ierr)
         CHKERRQ(ierr)
      end do
      do f=fStart,fEnd-1
         call PetscSectionSetDof(sec,f,1,ierr)
         CHKERRQ(ierr)
      end do
c
c     create Section (petsc way of setting DOFs)
c            
      call PetscSectionSetUp(sec,ierr)
      CHKERRQ(ierr)
      
      call PetscSectionGetStorageSize(sec,nsec,ierr)
      CHKERRQ(ierr)
c
      call PetscSectionView(sec,PETSC_VIEWER_STDOUT_WORLD,ierr)
      CHKERRQ(ierr)
c      
c
c     check DOFs at nodes ONLY
c
      write(*,*) ""
      write(*,*) "Nodes / DOFs"
      write(*,*) ""
      
      do ic=cStart,cEnd-1
         pp => pts         

         call DMPlexGetTransitiveClosure(dm,ic,PETSC_TRUE,pp,ierr)
         CHKERRQ(ierr)
         
         do j=1,2*npts,2
            ix = pp(j)

            if(vStart.le.ix .and. ix.lt.vEnd) then
               call PetscSectionGetOffset(sec,ix,ioff,ierr)
               CHKERRQ(ierr)
               call PetscSectionGetDof(sec,ix,idof,ierr)
               CHKERRQ(ierr)            
               write(*,'(A,I5,A,2I4)') "  Node",ix,"  dof",ioff,idof
            end if
         end do
         
         call DMPlexRestoreTransitiveClosure(dm,ic,PETSC_TRUE,pp,ierr)
         CHKERRQ(ierr)
c         
      end do      
c
c     fim
c
      write(*,*) ""
      write(*,*) "Finalizando"
      write(*,*) ""
c      
      call PetscSectionDestroy(sec,ierr)
      CHKERRQ(ierr)
c     
      call DMDestroy(dm,ierr)
      CHKERRQ(ierr)      
c
c     end
c      
      call PetscFinalize(ierr)
      CHKERRQ(ierr)
      
      end program testdmplex 
