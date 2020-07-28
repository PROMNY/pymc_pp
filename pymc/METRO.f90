subroutine METRO(RES, H, N, CP, T, U, ITER)

! sead var
  INTEGER :: ns, ival(8), v(3)
  INTEGER, ALLOCATABLE :: seed(:)

! in var
  INTEGER :: N,ITER
  REAL*8 :: CP,T,U, H(N,N)
  !f2py intent(in) N, ITER, CP, T, U, H

! out var
  REAL*8 :: E0, RES(N,N)
  !f2py intent(out) RES

! other var
  INTEGER :: I,S1,S2,SIZE,OK,MC
  REAL*8 :: P,BE,DET, TT(N,N), E
  REAL*8, DIMENSION(:), ALLOCATABLE :: WORK,EIGE

! sead set
  call date_and_time(values=ival)
  v(1) = ival(8) + 2048*ival(7)
  v(2) = ival(6) + 64*ival(5)     ! value(4) isn't really 'random'
  v(3) = ival(3) + 32*ival(2) + 32*8*ival(1)
  call random_seed(size=ns)
  allocate(seed(ns))
  call random_seed()   ! Give the seed an implementation-depENDent kick
  call random_seed(get=seed)
  do i=1, ns
     seed(i) = seed(i) + v(mod(i-1, 3) + 1)
  ENDdo
  call random_seed(put=seed)
  deallocate(seed)

! main

  !WRITE(*,*)"T=", T, " CP=",cp, " N=", N, " U=", U
  SIZE=N
  ALLOCATE(WORK(3*SIZE))
  ALLOCATE(EIGE(SIZE))

  be=1.0/T


  TT=H
  call DSYEV('N', 'U', SIZE, TT, SIZE, EIGE, work, size*3, ok)
  E0=0

  DO I=1,SIZE
      E0=E0+LOG(1+EXP(-BE*(EIGE(I)-CP)))
  ENDDO
  E0=-T*E0

! main loop
  DO MC=1,ITER
    DO
      CALL RANDOM_NUMBER(P)
      S1=INT(P*SIZE)+1
      IF (abs(H(S1,S1))>0.001) THEN
        EXIT
      ENDIF
    END DO


    DO
      CALL RANDOM_NUMBER(P)
      S2=INT(P*SIZE)+1
      IF (H(S2,S2)==0) THEN
        EXIT
      ENDIF
    END DO

    TT=H
    TT(S1,S1)=0
    TT(S2,S2)=U

    call DSYEV('N', 'U', SIZE, TT, SIZE, EIGE, work, size*3, ok)
    E=0

    DO I=1,SIZE
        E=E+LOG(1+EXP(-BE*(EIGE(I)-CP)))
    ENDDO
    E=-T*E

    DET=E-E0

    IF (DET<0) THEN
      H(S1,S1)=0
      H(S2,S2)=U
      E0=E
    ELSE
      CALL RANDOM_NUMBER(P)
      IF (EXP((-DET)*be)>=P) THEN
        H(S1,S1)=0
        H(S2,S2)=U
        E0=E
      END IF
    END IF
  ENDDO
  RES=H

end subroutine
