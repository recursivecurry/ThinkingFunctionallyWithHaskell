module Main where

import qualified Data.List as L

type Matrix a = [Row a]

type Row a = [a]

type Grid = Matrix Digit
type Digit = Char

digits :: [Char]
digits = ['1' .. '9']

blank :: Digit -> Bool
blank = (== '0')

solve :: Grid -> [Grid]
solve = filter valid . completions

completions :: Grid -> [Grid]
completions = expand . many prune . choices

many :: (Eq a) => (a -> a) -> a -> a
many f x = let y = f x in if x == y then x else many f y

choices :: Grid -> Matrix [Digit]
choices = map (map choice)

choice :: Digit -> [Digit]
choice d = if blank d then digits else [d]

cp :: [[a]] -> [[a]]
cp [] = [[]]
cp (xs:xss) = [x:ys | x <- xs, ys <- cp xss]

expand :: Matrix [Digit] -> [Grid]
expand = cp . map cp

valid :: Grid -> Bool
valid g = all nodups (rows g) &&
          all nodups (cols g) &&
          all nodups (boxs g)

nodups :: (Eq a) => [a] -> Bool
nodups [] = True
nodups (x:xs) = all (/=x) xs && nodups xs

rows :: Matrix a -> Matrix a
rows = id

cols :: Matrix a -> Matrix a
cols [xs] = [[x]|x<-xs]
cols (xs:xss) = zipWith (:) xs (cols xss)

boxs :: Matrix a -> Matrix a
boxs = map ungroup . ungroup . map cols . group . map group 

group :: [a] -> [[a]]
group [] = []
group xs = take 3 xs : group (drop 3 xs)

ungroup :: [[a]] -> [a]
ungroup = concat

prune :: Matrix [Digit] -> Matrix [Digit]
prune = rows . (map pruneRow) . rows . cols . (map pruneRow) . cols . boxs . (map pruneRow) . boxs

pruneRow :: (Eq a) => Row [a] -> Row [a]
pruneRow row = map (remove fixed) row
               where fixed = [d | [d] <- row]

remove :: (Eq a) => [a] -> [a] -> [a]
remove ds [x] = [x]
remove ds xs = filter (`notElem` ds) xs

all' :: (a -> Bool) -> [a] -> Bool
all' _ [] = True
all' f (x:xs) = f x && all' f xs

notElem' :: (Eq a) => a -> [a] -> Bool
notElem' x xs = all (/=x) xs

expand1 :: Matrix [Digit] -> [Matrix [Digit]]
expand1 rows = [rows ++ [row1 ++ [c]:row2] ++ rows2 | c <- cs]
               where
                 (rows1, row:rows2) = break (any (not . single)) rows
                 (row1, cs:row2) = break (not . single) row

single :: [a] -> Bool
single [_] = True
single _ = False

minimum :: (Ord a) => [a] -> a
minimum = foldr1 (\x y -> if x < y then x else y)
