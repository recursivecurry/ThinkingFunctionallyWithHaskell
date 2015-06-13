module Main where

import Data.List

mean :: [Float] -> Float
mean xs = sum xs / fromIntegral (length xs)

sumlen :: Num a => [a] -> (a, Int)
sumlen = foldl' g (0,0)
         where g (s,n) x = (s+x,n+1)

mean' [] = 0
mean' xs = s / fromIntegral n
          where (s,n) = sumlen xs

sumlen' = foldl' f (0, 0)
          where f (s, n) x = s `seq` n `seq` (s+x, n+1)

mean'' [] = 0
mean'' xs = s / fromIntegral n
           where (s,n) = sumlen' xs

main = do print $ mean [1, 3, 100000]
