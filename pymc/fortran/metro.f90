
subroutine metropolis(TM,N, res, m)

  INTEGER N
  REAL*8:: TM(N,N), RES(N)
  INTEGER :: I,S1,S2,SIZE,OK,MC, M
  REAL*8, DIMENSION(:,:),ALLOCATABLE :: TT
  REAL*8, DIMENSION(:),ALLOCATABLE :: WORK,EIGE



  !f2py intent(in) TM, N
  !f2py intent(out) RES, M

  m=n


  ALLOCATE(WORK(3*N))
  ALLOCATE(EIGE(N))
  ALLOCATE(TT(N,N))
  !ALLOCATE(OUT(N))

  TT = TM
  call DSYEV('N', 'U', N, TM, N, EIGE, work, N*3, ok)
  res = eige
end subroutine
