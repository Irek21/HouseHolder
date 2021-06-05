for ((p=1;p<64;p*=2))
do
echo $p
mpisubmit.pl -p $p run --512
done
