
-- imports
import qualified Data.Char as Char
import qualified Data.List as List

type Text = [Char]
type Word = [Char]

-- map :: (a->b) -> [a] -> [b] from Prelude
-- toLower :: Char -> Char from Data.Char
-- map toLower :: Text -> Text

sortWords :: [Word] -> [Word]
sortWords = List.sort

countRuns :: [Word] -> [(Int,Word)]
countRuns [] = []
countRuns (x:xs) = (hl, x) : countRuns t
                   where hl = 1 + (length . takeWhile (\v -> x==v)) xs
                         t = dropWhile (\v -> x==v) xs
                         
sortRuns :: [(Int,Word)] -> [(Int,Word)]
sortRuns = List.reverse . List.sort

-- take :: Int -> [a] -> [a]

showRun :: (Int,Word) -> String
showRun (x,y) = concat [y, "\t", show(x), "\n"]

-- map showRun :: [(Int,Word)] -> [String]

-- concat :: [[a]] -> [a] from Prelude
commonWords :: Int -> Text -> String
commonWords n = concat . map showRun . take n . 
                sortRuns . countRuns . sortWords . 
                words . map Char.toLower

sample = "cc ba aa ba cc cc"

putStrLn $ commonWords 2 sample

convert :: Int -> String
convert = convert6

units, teens, tens :: [String]
units = ["zero","one","two","three","four","five","six","seve","eight","nine"]
teens = ["ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
tens = ["twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

convert1 :: Int -> String
convert1 n = units !! n

digits2 :: Int -> (Int,Int)
digits2 n = (div n 10, mod n 10)
-- You can use infix version of div and mod by using `
-- digits2 n = (n `div` 10, n `mod` 10)
-- You can also use divMod
-- digits2 n = divMod n 10

convert2 :: Int -> String
convert2 = combine2 . digits2

combine2 :: (Int,Int) -> String
combine2 (t,u)
  | t==0 = units!!u
  | t==1 = teens!!u
  -- | 2<=t && u==0 = tens!!(t-2)
  | u==0 = tens!!(t-2)
  -- | 2<=t && u/=0 = tens!!(t-2) ++ "-" ++ units!!u
  | otherwise = tens!!(t-2) ++ "-" ++ units!!u

-- yet another way of writing convert2
convert2' :: Int -> String
convert2' n
  | t==0 = units!!u
  | t==1 = teens!!u
  | u==0 = tens!!(t-2)
  | otherwise = tens!!(t-2) ++ "-" ++ units!!u
  where (t,u) = (n `div` 10, n `mod` 10)

convert3 :: Int -> String
convert3 n
  | h==0 = convert2 t
  | n==0 = units!!h ++ " hundred"
  | otherwise = units!!h ++ " hundred and " ++ convert2 t
  where (h,t) = (n `div` 100, n `mod` 100)

convert6 :: Int -> String
convert6 n
  | m==0 = convert3 h
  | h==0 = convert3 m ++ " thousand"
  | otherwise = convert3 m ++ " thousand" ++ link h ++ convert3 h
  where (m,h) = (n `div` 1000, n `mod` 1000)

link :: Int -> String
link h = if h < 100 then " and " else " "
-- We could also have used guarded equations
-- link h | h < 100 = " and "
--        | otherwise = " "

convert 308000
convert 369027

double :: Integer -> Integer
double x = 2 * x

map double [1,4,4,3]
map (double . double) [1,4,4,3]
map double []

sample2 = [1,2,3,4]
sample3 = [[1,2], [3,4], [5]]

(List.sum . map double) sample2 == (double . List.sum) sample2
(List.sum . map List.sum) sample3 == (List.sum . concat) sample3
(List.sum . List.sort) sample2 == List.sum sample2

(words . map Char.toLower) sample
(map (map Char.toLower) . words) sample

anagrams :: Int -> [Word] -> String
anagrams n = List.concat . List.intersperse "\n" . map showItem . groupItems . sortItems . map normalize . take n

normalize :: Word -> (Word,Word)
normalize w = (List.sort w, w)

sortItems :: [(Word,Word)] -> [(Word,Word)]
sortItems = List.sort

groupItems :: [(Word,Word)] -> [(Word,[Word])]
groupItems [] = []
groupItems xxs@((n,w):xs) = (n, hl) : groupItems t
                   where hl = (map snd . takeWhile (\(v,vs) -> n==v)) xxs
                         t = dropWhile (\(v,vs) -> n==v) xs

showItem :: (Word,[Word]) -> String
showItem (x,xs) = show . List.concat $ x:": ":(List.intersperse "," xs)

putStr $ anagrams 5 ["abc", "bca", "cab", "def", "fed"]

song :: Int -> String
song n  = if n==0 then ""
          else song (n-1) ++ "\n" ++ verse n

verse :: Int -> String
verse n = line1 n ++ line2 n ++ line3 n ++ line4 n

number = ["","one","two","three","four",
          "five","six","seven","eight","nine"]

man :: Int -> String
man n = number !! n ++ if n == 1 then " man" else " men"

mans :: Int -> String
mans n = (concat . List.intersperse ", " . map man . reverse)  [1..n]

cap :: String -> String
cap [] = []
cap (x:xs) = Char.toUpper x : xs

line1 :: Int -> String
line1 n = cap $ man n ++ " went to mow\n"

line2 :: Int -> String
line2 n = "Went to mow a meadow\n"

line3 :: Int -> String
line3 n = cap $ mans n ++ " and his dog\n"

line4 :: Int -> String
line4 n = "Went to mow a meadow\n"

putStr $ song 1
putStr $ song 2
