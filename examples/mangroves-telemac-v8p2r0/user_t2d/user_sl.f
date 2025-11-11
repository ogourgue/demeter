!                   ******************
                    SUBROUTINE USER_SL
!                   ******************
!
     &(I , N, SL)
!
!***********************************************************************
! TELEMAC2D
!***********************************************************************
!
!brief    USER PRESCRIBES THE FREE SURFACE ELEVATION FOR LEVEL IMPOSED
!+                LIQUID BOUNDARIES.
!
!history  J-M HERVOUET (LNHE)
!+        17/08/1994
!+        V6P0
!+
!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!| I              |-->| NUMBER OF LIQUID BOUNDARY
!| N              |-->| GLOBAL NUMBER OF POINT
!|                |   | IN PARALLEL NUMBER IN THE ORIGINAL MESH
!| SL             |<->| FREE SURFACE ELEVATION
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
      USE BIEF
      USE DECLARATIONS_SPECIAL
      USE DECLARATIONS_TELEMAC
      USE DECLARATIONS_TELEMAC2D
      USE INTERFACE_TELEMAC2D, EX_USER_SL => USER_SL, EX_SL => SL
!
      IMPLICIT NONE
!
!+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
!
      INTEGER, INTENT(IN) :: I,N
      DOUBLE PRECISION, INTENT(INOUT) :: SL
!
!+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
!
! ----------------------------------------------------------------------
! START TIGER
!
      INTEGER NH
      DOUBLE PRECISION TIR, PI, TM2, SLRR
!
!     TIDAL RANGE (M)
      TIR = 5.D0
!
!     PI
      PI = 4.D0 * ATAN(1.D0)
!
!     M2 TIDAL PERIOD (S)
      TM2 = (12.D0 * 60.D0 + 25.D0) * 60.D0
!
!     SEA LEVEL RISE RATE (M/YR)
      SLRR = 5.D-3

!     NUMBER OF TIDAL CYCLES IN ONE MORPHOLOGICAL YEAR
      NH = 1.D0
!
!     SURFACE LEVEL AT THE BOUNDARY
      SL = .5D0 * TIR * SIN(2.D0 * PI * AT / TM2)
     &   + SLRR * AT0 / (NH * TM2)
!
! END TIGER
! ----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
!
      RETURN
      END
