!                   *****************
                    SUBROUTINE DRAGFO
!                   *****************
!
     &(FUDRAG,FVDRAG)
!
!***********************************************************************
! TELEMAC2D   V6P2                                   21/08/2010
!***********************************************************************
!
!brief    ADDS THE DRAG FORCE OF VERTICAL STRUCTURES IN THE
!+                MOMENTUM EQUATION.
!code
!+  FU IS THEN USED IN THE EQUATION AS FOLLOWS :
!+
!+  DU/DT + U GRAD(U) = - G * GRAD(FREE SURFACE) +..... + FU_IMP * U
!+
!+  AND THE TERM FU_IMP * U IS TREATED IMPLICITLY.
!
!warning  USER SUBROUTINE
!
!history  J-M HERVOUET
!+        01/03/1990
!+        V5P2
!+
!
!history  N.DURAND (HRW), S.E.BOURBAN (HRW)
!+        13/07/2010
!+        V6P0
!+   Translation of French comments within the FORTRAN sources into
!+   English comments
!
!history  N.DURAND (HRW), S.E.BOURBAN (HRW)
!+        21/08/2010
!+        V6P0
!+   Creation of DOXYGEN tags for automated documentation and
!+   cross-referencing of the FORTRAN sources
!
!history  J,RIEHME (ADJOINTWARE)
!+        November 2016
!+        V7P2
!+   Replaced EXTERNAL statements to parallel functions / subroutines
!+   by the INTERFACE_PARALLEL
!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!| FUDRAG         |<--| DRAG FORCE ALONG X
!| FVDRAG         |<--| DRAG FORCE ALONG Y
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
      USE BIEF
      USE DECLARATIONS_TELEMAC2D
!
      USE DECLARATIONS_SPECIAL
      USE INTERFACE_PARALLEL, ONLY : P_SUM
      IMPLICIT NONE
!
!+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
!
      TYPE(BIEF_OBJ), INTENT(INOUT) :: FUDRAG,FVDRAG
!
!+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
!
! START TIGER
!
      INTEGER I
      DOUBLE PRECISION CD, M, D, K, UNORM, CVEG
!
!     BULK DRAG COEFFICIENT
      CD = 5.D0
!
!     STEM DENSITY
      M = 658.D0
!
!     STEM DIAMETER
      D = 3.D-3
!
!     STEM HEIGHT
      K = 5.9D-1
!
!     COMPUTE DRAG FORCE AT EACH GRID POINT
      DO I = 1, NPOIN
!
!       FLOW VELOCITY NORM AND DIRECTION
        UNORM = SQRT(UN%R(I) * UN%R(I) + VN%R(I) * VN%R(I))
!
!       EQUIVALENT CHEZY COEFFICIENT DUE TO VEGETATION DRAG
        IF(HN%R(I) <= K) THEN
          CVEG = SQRT((2 * GRAV) / (CD * COV%R(I) * M * D * HN%R(I)))
        ELSE
          CVEG = SQRT((2 * GRAV) / (CD * COV%R(I) * M * D * K))
     &         + SQRT(GRAV) / KARMAN * LOG(HN%R(I) / K)
        ENDIF
!
!       DRAG FORCE
        FUDRAG%R(I) = -(GRAV * UNORM) / (HN%R(I) * CVEG * CVEG)
        FVDRAG%R(I) = -(GRAV * UNORM) / (HN%R(I) * CVEG * CVEG)
!
      ENDDO
!
!
! END TIGER
!
!-----------------------------------------------------------------------
!
      RETURN
      END
