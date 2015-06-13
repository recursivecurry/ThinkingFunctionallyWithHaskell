module Foo where

foo1 n = sum (take n primes)
         where
		   primes = [x | x <- [2..], divisors x == [x]]
		   divisors x = [d | d <- [2..x], x `mod` d == 0]

foo2 n = sum (take n primes)
primes = [x | x <- [2..], divisors x == [x]]
divisors x = [d | d <- [2..x], x `mod` d == 0]

foo3 = \n -> sum (take n primes)
       where
		   primes = [x | x <- [2..], divisors x == [x]]
		   divisors x = [d | d <- [2..x], x `mod` d == 0]

main = undefined
