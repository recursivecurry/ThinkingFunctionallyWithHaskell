cp [] = [[]]
cp (xs:xss) = [x:ys | x <- xs, ys <- cp xss]

cp' = foldr op [[]]
      where op xs yss = [x:ys | x <- xs, ys <- yss]

cp'' [] = [[]]
cp'' (xs:xss) = [x:ys | x <- xs, ys <- yss]
               where yss = cp xss
