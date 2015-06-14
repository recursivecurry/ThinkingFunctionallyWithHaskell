
# Chapter 7 Efficiency

미국의 컴퓨터 과학자인 Alan Perlis가 애기하기를...

> A functional programmer was someone who knew the value of everything and the cost of nothing.

functional programming의 **성능**과 **효율성**은?

# 7.1 Lazy Evaluation


## Lazy


    sqr x = x * x
    
    sqr (sqr (3+4))


    2401


<s>위 expression에 대해서 아래와 같이 lazy하게 reduction 된다.</s>

    sqr (sqr (3+4))
    = sqr (3+4) * sqr (3+4)
    = ((3+4)*(3+4)) * ((3+4)*(3+4))
    = ...
    = 2401

lazy evaluation은 아래와 같이 reduction된다.

    sqr (sqr (3+4))
    = let x = sqr (3+4) in sqr x
    = let y = 3+4 in
      let x = sqr y in sqr x
    = let y = 7 in
      let x = sqr y in sqr x
    = let x = sqr 7 in sqr x
    = let x = 7 * 7 in sqr x
    = let x = 49 in sqr x
    = 49 * 49
    = 2401


> <s>Lazy evaluation arguments는 필요한 시점에 단 한 번만 evaluate 된다.</s>

보다 정확하게 얘기하면

> Lazy evaluation에서 arguments는 **필요한 시점**에 단 **한 번만** **head normal form** 으로 evaluate 된다.

## Head Normal Form

(Weak) Head Normal Form은 top-level이 아래로 구성되어 있다.

 * data constructor
 * fully reduced lambda abstraction (weak: any lambda abstraction)

NF는 HNF이나 HNF가 NF는 아니다.

 * NF: 42, (2,"hello"), \x -> (x+1), e1, e2
 * HNF: (1+2,2+3), \x -> 2+2, 'h':("e"++"llo"), x, (e1, e2)
 * ???: 1+1, (\x -> x+1) 2

'sqr (head xs)'의 evaluation

    sqr (head xs)
    = let a = head xs in sqr a
    = let b = xs in
      let a = head xs in sqr a
    = let b = y:ys in
      let a = head xs in sqr a
    = let a = head (y:ys) in sqr a
    = let a = y in sqr a
    = sqr y
    = y * y

## Common Subexpression elimination

    subseqs (x:xs) = subseqs xs ++ map (x:) (subseqs xs)
    subseqs' (x:xs) = xss ++ map (x:) xss
                      where xss = subseqs' xs

subseqs는 'subseqs xs'가 2번 evaluate된다. 이것을 where로 분리하는 경우 1번만 연산할 수 있으나 **haskell은 common subexpression elimination 을 자동으로 하지 않는다.**
추가 공간을 사용하여 시간을 최적화하는 것이므로 프로그래머가 결정해야하는 사항임.

## Binding


    foo1 n = sum (take n primes)
             where
               primes = [x | x <- [2..], divisors x == [x]]
               divisors x = [d | d <- [2..x], x `mod` d == 0]
    foo1 100


    24133



    foo2 n = sum (take n primes)
    primes = [x | x <- [2..], divisors x == [x]]
    divisors x = [d | d <- [2..x], x `mod` d == 0]
    
    foo2 100


    24133



    foo3 = \n -> sum (take n primes)
           where
             primes = [x | x <- [2..], divisors x == [x]]
             divisors x = [d | d <- [2..x], x `mod` d == 0]
    
    foo3 100


    24133


primes, divisors의 binding

 * foo1: 'foo1 n'에 bind 됨
 * foo2: global에 bind 됨
 * foo3: 'foo3'에 bind 됨

foo2, foo3의 경우 evaluation 결과를 재사용하나 공간 사용량이 증가함

# 7.2. Controlling space

## lazy and space

lazy evaluation

    sum [1..1000]
    = foldl (+) 0 [1..1000]
    = foldl (+) (0+1) [2..1000]
    = foldl (+) ((0+1)+2) [3..1000]
    = ...
    = 500500

eager evaluation

    sum [1..1000]
    = foldl (+) 0 [1..1000]
    = foldl (+) (0+1) [2..1000]
    = foldl (+) 1 [2..1000]
    = foldl (+) (1+2) [3..1000]
    = ...
    = 500500

**space를 제어하기 위해서는 eager evaluation을 lazy evaluation과 함께 사용하는 것이 좋다.**

## eager evaluation


    -- seq :: a -> b -> b
    
    -- Data.List.foldl'
    foldl' :: (b -> a -> b) -> b -> [a] -> b
    foldl' f e [] = e
    foldl' f e (x:xs) = y `seq` foldl' f y xs
                        where y = f e x
    
    foldl' (+) 0 [1..10]


    55


Data\.List\.sum과 foldl'은 위 방식으로 구현됨.

lazy와 eager 구현은 strict function에 대해서는 동일하다. (f &perp; = &perp; 라면, f는 strict하다.)

## mean 예제


    mean [] = 0
    mean xs = sum xs / fromIntegral (length xs)
    
    mean [1..10]


    5.5


xs의 순회가 2번 발생한다. tupling을 사용하여 1번 순회로 합과 길이를 구함.


    sumlen :: [Float] -> (Float,Int)
    sumlen = foldr f (0,0)
                 where f x (s,n) = (s+x,n+1)
    
    sumlen [1..10]


    (55.0,10)



    mean' [] = 0
    mean' xs = s / fromIntegral n
              where (s,n) = sumlen xs
    mean' [1..10]


    5.5


space leak을 제거하기 위해서 foldl'을 사용


    sumlen = foldl' g (0,0)
             where g (s,n) x = (s+x,n+1)
    
    sumlen [1..10]


    (55,10)


HNF인 (s+x,n+1)에 eager evaluation 적용


    sumlen = foldl' f (0,0)
             where f (s,n) x = s `seq` n `seq` (s+x,n+1)
    
    sumlen [1..10]


    (55,10)


two more application operators

 * $: lazy evaluation
 * \$\!: eager evaluation

# 7.3 Controlling time

## tips for time

eager evalution을 사용하여 쉽게 제어할 수 있으나 속도는 그렇지 못 하다.

GHC 문서가 제시하는 속도향상의 키 포인트 3가지

1. GHC Profiling 도구를 사용해라
2. 알고리즘을 개선해라
3. 가능하면 제공되는 라이브러리를 사용하라. (매우 최적화되어 있고, 컴파일되어 있음.)

추가적인 팁 2가지

* 'Strict funcions are your dear friends': 보통 eager evaluation이 lazy evaluation보다 overhead가 적다.
* 요구사항을 만족하는 최적의 type을 명시적으로 사용할 것: Integer보다는 Int가 좋다.

이러한 방법들은 asymptotic time complexity을 변환시키지 못 하므로 효과가 작다. 그러나 잘못된 코드는 asymptotic complexity에 영향을 줄 수도 있다.

## cartesian product


    cp [] = [[]]
    cp (xs:xss) = [x:ys | x <- xs, ys <- cp xss]
    
    cp [[1,2],[3,4],[5,6]]


    [[1,3,5],[1,3,6],[1,4,5],[1,4,6],[2,3,5],[2,3,6],[2,4,5],[2,4,6]]



    cp' = foldr op [[]]
          where op xs yss = [x:ys | x <- xs, ys <- yss]
    
    cp' [[1,2],[3,4],[5,6]]


    [[1,3,5],[1,3,6],[1,4,5],[1,4,6],[2,3,5],[2,3,6],[2,4,5],[2,4,6]]


cp보다 cp'의 evaluation time이 훨씬 적다. cp는 cp xss를 xs의 길이만큼 반복 계산한다. cp의 list comprehension을 제거하면 아래 형태로 이해가 쉬워진다.


    cp [] = [[]]
    cp (xs:xss) = concat (map f xs)
                  where f x = [x:ys | ys <- cp xss]
    
    cp [[1,2],[3,4],[5,6]]


    [[1,3,5],[1,3,6],[1,4,5],[1,4,6],[2,3,5],[2,3,6],[2,4,5],[2,4,6]]


아래 cp''는 cp'와 동일한 성능을 가진다.


    cp'' [] = [[]]
    cp'' (xs:xss) = [x:ys | x <- xs, ys <- yss]
                    where yss = cp xss
    
    cp'' [[1,2],[3,4],[5,6]]


    [[1,3,5],[1,3,6],[1,4,5],[1,4,6],[2,3,5],[2,3,6],[2,4,5],[2,4,6]]


# 7.4 Analysing time

* 시간 복잡도는 expression의 속성이지 value의 속성은 아니다.
* GHCi에서는 reduction step이 아닌 소요시간만을 측정한다. reduction 단계는 소요 시간과 반드시 일치하지는 않는다.
* 상황에 따라서 다른 측정 방법이 필요하다. 예를 들면 concat xss의 경우 n보다는 (m,n)이 적합하다.
* 시간 복잡도 측정은 eager evaluation으로 행한다. lazy evaluation은 측정이 어려우며 일반적으로 eager evaluation의 time이 lazy evaluation의 time 보다 tight boundary를 가진다.

## Concat


    concat xss = foldr (++) [] xss
    
    concat [[1,2],[3,4]]


<style>/*
Custom IHaskell CSS.
*/

/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
    display: block;
    padding-bottom: 1.3em;
    padding-left: 0.4em;
}
.hoogle-code {
    display: block;
    font-family: monospace;
    white-space: pre;
}
.hoogle-text {
    display: block;
}
.hoogle-name {
    color: green;
    font-weight: bold;
}
.hoogle-head {
    font-weight: bold;
}
.hoogle-sub {
    display: block;
    margin-left: 0.4em;
}
.hoogle-package {
    font-weight: bold;
    font-style: italic;
}
.hoogle-module {
    font-weight: bold;
}
.hoogle-class {
    font-weight: bold;
}

/* Styles used for basic displays */
.get-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    display: block;
    white-space: pre-wrap;
}

.show-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    margin-left: 1em;
}

.mono {
    font-family: monospace;
    display: block;
}

.err-msg {
    color: red;
    font-style: italic;
    font-family: monospace;
    white-space: pre;
    display: block;
}

#unshowable {
    color: red;
    font-weight: bold;
}

.err-msg.in.collapse {
  padding-top: 0.7em;
}

/* Code that will get highlighted before it is highlighted */
.highlight-code {
    white-space: pre;
    font-family: monospace;
}

/* Hlint styles */
.suggestion-warning { 
    font-weight: bold;
    color: rgb(200, 130, 0);
}
.suggestion-error { 
    font-weight: bold;
    color: red;
}
.suggestion-name {
    font-weight: bold;
}
</style> <div class="suggestion-name" style="clear:both;">Eta reduce</div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Found:</div>  <div class="highlight-code" id="haskell">concat xss = foldr (++) [] xss</div> </div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Why Not:</div>  <div class="highlight-code" id="haskell">concat = foldr (++) []</div> </div>  <div class="suggestion-name" style="clear:both;">Use concat</div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Found:</div>  <div class="highlight-code" id="haskell">foldr (++) []</div> </div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Why Not:</div>  <div class="highlight-code" id="haskell">concat</div> </div> 



    [1,2,3,4]



    concat' xss = foldl (++) [] xss
    
    concat' [[1,2],[3,4]]


<style>/*
Custom IHaskell CSS.
*/

/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
    display: block;
    padding-bottom: 1.3em;
    padding-left: 0.4em;
}
.hoogle-code {
    display: block;
    font-family: monospace;
    white-space: pre;
}
.hoogle-text {
    display: block;
}
.hoogle-name {
    color: green;
    font-weight: bold;
}
.hoogle-head {
    font-weight: bold;
}
.hoogle-sub {
    display: block;
    margin-left: 0.4em;
}
.hoogle-package {
    font-weight: bold;
    font-style: italic;
}
.hoogle-module {
    font-weight: bold;
}
.hoogle-class {
    font-weight: bold;
}

/* Styles used for basic displays */
.get-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    display: block;
    white-space: pre-wrap;
}

.show-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    margin-left: 1em;
}

.mono {
    font-family: monospace;
    display: block;
}

.err-msg {
    color: red;
    font-style: italic;
    font-family: monospace;
    white-space: pre;
    display: block;
}

#unshowable {
    color: red;
    font-weight: bold;
}

.err-msg.in.collapse {
  padding-top: 0.7em;
}

/* Code that will get highlighted before it is highlighted */
.highlight-code {
    white-space: pre;
    font-family: monospace;
}

/* Hlint styles */
.suggestion-warning { 
    font-weight: bold;
    color: rgb(200, 130, 0);
}
.suggestion-error { 
    font-weight: bold;
    color: red;
}
.suggestion-name {
    font-weight: bold;
}
</style> <div class="suggestion-name" style="clear:both;">Eta reduce</div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Found:</div>  <div class="highlight-code" id="haskell">concat' xss = foldl (++) [] xss</div> </div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Why Not:</div>  <div class="highlight-code" id="haskell">concat' = foldl (++) []</div> </div>  <div class="suggestion-name" style="clear:both;">Use concat</div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Found:</div>  <div class="highlight-code" id="haskell">foldl (++) []</div> </div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Why Not:</div>  <div class="highlight-code" id="haskell">concat</div> </div> 



    [1,2,3,4]


T(++)(n,m) = $Ө(n)$임

* concat: 길이 n의 리스트를 m회 (++) 함 $\Theta (mn)$
* concat': mn으로 증가하는 accumulator를 n회 (++) 함. $\Theta (m^2 n$)

concat은 foldr로 구현하는 것이 더 효율적임. foldl/foldr/foldl'이 적합한 경우가 다르다.

## subseq


    subseqs [] = [[]]
    subseqs (x:xs) = subseqs xs ++ map (x:) (subseqs xs)
    
    subseqs [1..3]


    [[],[3],[2],[2,3],[1],[1,3],[1,2],[1,2,3]]



    subseqs' [] = [[]]
    subseqs' (x:xs) = xss ++ map (x:) xss
                      where xss = subseqs' xs
    
    subseqs' [1..3]


    [[],[3],[2],[2,3],[1],[1,3],[1,2],[1,2,3]]


map (x:)는 $\Theta (2^n)$이므로 

* $T(subseqs)(n+1) = 2T(subseqs)(n) + \Theta (2^n) \rightarrow T(subseqs)(n) = \Theta (n 2^n)$
* $T(subseqs')(n+1) = T(subseqs')(n) + \Theta (2^n) \rightarrow T(subseqs')(n) = \Theta (2^n)$

subseqs'가 logarithmic factor로 더 빠르다. 속도가 중요한 경우 common subexpression elimination을 잘 활용하자.

## cartesian product


    cp [] = [[]]
    cp (xs:xss) = [x:ys | x <- xs, ys <- cp xss]
    
    cp [[1,2],[3,4]]


    [[1,3],[1,4],[2,3],[2,4]]



    cp' = foldr op [[]]
          where op xs yss = [x:ys | x <- xs, ys <- yss]
    
    cp' [[1,2],[3,4]]


    [[1,3],[1,4],[2,3],[2,4]]


cp에서 cp xss가 $\Theta (n^m)$ 이고 m회만큼 반복하게 된다.

* $T(cp)(m,n) = \Theta (m n^m)$
* $T(cp')(m,n) = \Theta (n^m)$

cp'가 logarithmic factor로 더 빠르다.

# 7.5 Accumulating Parameter

argument를 추가로 사용하여 속도를 향상시키는 것을 accumulating parameter라고 한다.

## reverse


    reverse [] = []
    reverse (x:xs) = reverse xs ++ [x]
    
    reverse [1..10]


    [10,9,8,7,6,5,4,3,2,1]


(++)가 $\Theta (n)$이고 n번 재귀를 하므로 T(reverse)(n)은 $\theta (n^2)$ 이다.


    revcat :: [a] -> [a] -> [a]
    revcat xs ys = reverse xs ++ ys
    
    revcat [1..10] []


    [10,9,8,7,6,5,4,3,2,1]


revcat은 accumulator를 사용해서 (++) 대신 (:) 을 사용한 것과 동일해짐. 따라서 $\theta (n)$ 임

    [x] ++ xs = x:xs

## length


    length :: [a] -> Int
    length [] = 0
    length (x:xs) = length xs + 1
    
    length [1..10]


    10



    lenplus :: [a] -> Int -> Int
    lenplus [] n = n
    lenplus (x:xs) n = lenplus xs (1+n)
    
    lenplus [1..10] 0


    10


time은 $\Theta (n)$으로 동일하다. length는 space가 $\Theta (n)$으로 증가하나 lenplus는 $\Theta (1)$ 이다. (haskell의 length는 lenplus처럼 구현됨.)

## tree


    data GenTree a = Node a [GenTree a]


    labels :: GenTree a -> [a]
    labels (Node x ts) = x:concat (map labels ts)
    
    labels (Node 1 [Node 2 [],Node 3 []])


<style>/*
Custom IHaskell CSS.
*/

/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
    display: block;
    padding-bottom: 1.3em;
    padding-left: 0.4em;
}
.hoogle-code {
    display: block;
    font-family: monospace;
    white-space: pre;
}
.hoogle-text {
    display: block;
}
.hoogle-name {
    color: green;
    font-weight: bold;
}
.hoogle-head {
    font-weight: bold;
}
.hoogle-sub {
    display: block;
    margin-left: 0.4em;
}
.hoogle-package {
    font-weight: bold;
    font-style: italic;
}
.hoogle-module {
    font-weight: bold;
}
.hoogle-class {
    font-weight: bold;
}

/* Styles used for basic displays */
.get-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    display: block;
    white-space: pre-wrap;
}

.show-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    margin-left: 1em;
}

.mono {
    font-family: monospace;
    display: block;
}

.err-msg {
    color: red;
    font-style: italic;
    font-family: monospace;
    white-space: pre;
    display: block;
}

#unshowable {
    color: red;
    font-weight: bold;
}

.err-msg.in.collapse {
  padding-top: 0.7em;
}

/* Code that will get highlighted before it is highlighted */
.highlight-code {
    white-space: pre;
    font-family: monospace;
}

/* Hlint styles */
.suggestion-warning { 
    font-weight: bold;
    color: rgb(200, 130, 0);
}
.suggestion-error { 
    font-weight: bold;
    color: red;
}
.suggestion-name {
    font-weight: bold;
}
</style> <div class="suggestion-name" style="clear:both;">Use concatMap</div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Found:</div>  <div class="highlight-code" id="haskell">concat (map labels ts)</div> </div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Why Not:</div>  <div class="highlight-code" id="haskell">concatMap labels ts</div> </div> 



    [1,2,3]


       1
      / \
     2   3 labels = "123"

> $T(labels)(1,k) = \Theta(1)$
> $T(labels)(h+1,k) = \Theta(1) + T(concat)(k,s) + T(map labels)(h,k)$

각 높이에서 k개의 subtree에 대해서 map labels가 수행되고 이것들을 concat해야됨. 

$$T(labels)(h+1,k) = \Theta(k^{h+1}) + k T(labels)(h,k)$$

따라서 $s = k^h$일때 $\Theta (s \log s)$ 임.


    labcat :: [GenTree a] -> [a] -> [a]
    labcat ts xs = concat (map labels ts) ++ xs
    
    labcat [Node 1 [Node 2 [],Node 3 []]] []


<style>/*
Custom IHaskell CSS.
*/

/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
    display: block;
    padding-bottom: 1.3em;
    padding-left: 0.4em;
}
.hoogle-code {
    display: block;
    font-family: monospace;
    white-space: pre;
}
.hoogle-text {
    display: block;
}
.hoogle-name {
    color: green;
    font-weight: bold;
}
.hoogle-head {
    font-weight: bold;
}
.hoogle-sub {
    display: block;
    margin-left: 0.4em;
}
.hoogle-package {
    font-weight: bold;
    font-style: italic;
}
.hoogle-module {
    font-weight: bold;
}
.hoogle-class {
    font-weight: bold;
}

/* Styles used for basic displays */
.get-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    display: block;
    white-space: pre-wrap;
}

.show-type {
    color: green;
    font-weight: bold;
    font-family: monospace;
    margin-left: 1em;
}

.mono {
    font-family: monospace;
    display: block;
}

.err-msg {
    color: red;
    font-style: italic;
    font-family: monospace;
    white-space: pre;
    display: block;
}

#unshowable {
    color: red;
    font-weight: bold;
}

.err-msg.in.collapse {
  padding-top: 0.7em;
}

/* Code that will get highlighted before it is highlighted */
.highlight-code {
    white-space: pre;
    font-family: monospace;
}

/* Hlint styles */
.suggestion-warning { 
    font-weight: bold;
    color: rgb(200, 130, 0);
}
.suggestion-error { 
    font-weight: bold;
    color: red;
}
.suggestion-name {
    font-weight: bold;
}
</style> <div class="suggestion-name" style="clear:both;">Use concatMap</div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Found:</div>  <div class="highlight-code" id="haskell">concat (map labels ts)</div> </div>  <div class="suggestion-row" style="float: left;"> <div class="suggestion-error">Why Not:</div>  <div class="highlight-code" id="haskell">concatMap labels ts</div> </div> 



    [1,2,3]


    labcat (Node x us:vs) xs
    = {definition}
      concat (map labels (Node x us:vs)) ++ xs
    = {definitions}
      labels (Node x us) ++ concat (map labels vs) ++ xs
    = {definiton}
      x:concat (map labels us) ++ concat (map labels vs) ++ xs
    = {definition of labcat}
      x:concat (map labels us) ++ labcat vs xs
    = {definition of labcat (again)}
      x:labcat us (labcat vs xs)


    labels' t = labcat' [t] []
    
    labcat' [] xs = xs
    labcat' (Node x us:vs) xs = x:labcat' us (labcat' vs xs)
    
    labcat' [Node 1 [Node 2 [],Node 3 []]] []


    [1,2,3]


* $T(labcat)(1,k,n) = \Theta (n)$
* $T(labcat)(h,k,n) = \Theta (k^h n)$
* tree size $s = k^h$

따라서 $T(labels)(h,k) = T(labcat)(h,k,1) = \Theta (s)$

# 7.6 Tupling

# fibonacci



    fib :: Int -> Integer
    fib 0 = 0
    fib 1 = 1
    fib n = fib (n-1) + fib (n-2)
    
    fib 10


    55


$T(fib)(n) = \Theta (\phi ^ n)$ 이며 golden ratio $\phi = (1 + \sqrt{5}) / 2$


    fib' 0 = (0,1)
    fib' n = (b,a+b) where (a,b) = fib' (n-1)
    
    fst $ fib' 10


    55


fib는 exponential time이나 fib'는 linear time임.

## general law for tupling

    (foldr f a xs, foldr g b xs) = foldr h (a,b) xs
    h x (y,z) = (f x y, g x z)

## leaf-labelled binary tree building


    data BinTree a = Leaf a | Fork (BinTree a) (BinTree a) deriving (Show)


    halve xs = (take m xs, drop m xs)
               where m = length xs `div` 2


    build :: [a] -> BinTree a
    build [x] = Leaf x
    build xs = Fork (build ys) (build zs)
               where (ys,zs) = halve xs
    
    build [1,2,3]


    Fork (Leaf 1) (Fork (Leaf 2) (Leaf 3))


halve는 xs에 대해서 총 3회 순회를 하도록 구현되어 있어서 비효율적임.
$$ T(build)(n) = \Theta(n \log n)$$


    build2 :: Int -> [a] -> (BinTree a,[a])
    build2 n xs = (build (take n xs), drop n xs)
    
    build' xs = fst (build2 (length xs) xs)
    
    build' [1,2,3]


    Fork (Leaf 1) (Fork (Leaf 2) (Leaf 3))


build2는 drop n xs 부분도 반환하여 tupling하도록 함


    build2 1 xs = (Leaf (head xs),tail xs)
    build2 n xs = (Fork (build (take m (take n xs)))
                        (build (drop m (take n xs))),
                   drop n xs)
                  where m = n `div` 2
    
    build2 (length [1,2,3]) [1,2,3]


    (Fork (Leaf 1) (Fork (Leaf 2) (Leaf 3)),[])


    take m . take n = take m
    drop m . take n = take (n-m) . drop m


    build2 1 xs = (Leaf (head xs),tail xs)
    build2 n xs = (Fork (build (take m xs))
                        (build (take (n-m) (drop m xs))),
                   drop n xs)
                  where m = n `div` 2
    
    build2 (length [1,2,3]) [1,2,3]


    (Fork (Leaf 1) (Fork (Leaf 2) (Leaf 3)),[])



    build2 1 xs = (Leaf (head xs),tail xs)
    build2 n xs = (Fork u v, drop n xs)
                  where (u,xs') = build2 m xs
                        (v,xs'') = build2 (n-m) xs'
                        m = n `div` 2

    xs'' = drop  (n-m) xs'
         = drop (n-m) (drop m xs)
         = drop n xs


    build2 1 xs = (Leaf (head xs),tail xs)
    build2 n xs = (Fork u v, xs'')
                  where (u,xs') = build2 m xs
                        (v,xs'') = build2 (n-m) xs'
                        m = n `div` 2
                        
    fst $ build2 (length [1,2,3]) [1,2,3]


    Fork (Leaf 1) (Fork (Leaf 2) (Leaf 3))


$T(build2)(1) = \Theta (1)$

$T(build2)(n) = T(build2)(m) + T(build2)(n-m) + \Theta (1)$

따라서 $T(build2)(n) = \Theta (n)$ 임. logarithmic factor로 개선됨.

# 7.7 Sorting

## Merge sort


    merge :: (Ord a) => [a] -> [a] -> [a]
    merge [] ys = ys
    merge xs [] = xs
    merge xs'@(x:xs) ys'@(y:ys)
          | x <= y    = x:merge xs ys'
          | otherwise = y:merge xs' ys


    halve xs = (take m xs,drop m xs)
               where m = length xs `div` 2


    msort :: (Ord a) => [a] -> [a]
    msort [] = []
    msort [x] = [x]
    msort xs = merge (msort ys) (msort zs)
               where (ys,zs) = halve xs
    
    msort [10,8..1]


    [2,4,6,8,10]


1\. take, drop을 splitAt으로 변경하여 순회를 줄인다. Prelude.splitAt은 아래와 같이 tupling을 이용하여 구현됨.


    splitAt :: Int -> [a] -> ([a],[a])
    splitAt 0 xs = ([],xs)
    splitAt n [] = ([],[])
    splitAt n (x:xs) = (x:ys,zs)
                       where (ys,zs) = splitAt (n-1) xs
    
    halve xs = splitAt (length xs `div` 2) xs
    
    msort :: (Ord a) => [a] -> [a]
    msort [] = []
    msort [x] = [x]
    msort xs = merge (msort ys) (msort zs)
               where (ys,zs) = halve xs
    
    msort [10,8..1]


    [2,4,6,8,10]


2\. 이전의 tree build와 유사하게 sort를 tupling 한다.


    msort2 0 xs = ([],xs)
    msort2 1 xs = ([head xs], tail xs)
    msort2 n xs = (merge ys zs, xs'')
                  where (ys,xs') = msort2 m xs
                        (zs,xs'') = msort2 (n-m) xs'
                        m = n `div` 2
    
    fst $ msort2 (length [10,8..1]) [10,8..1]


    [2,4,6,8,10]


3\. halve를 다른 방법으로 사람처럼 해봄


    halve2 [] = ([],[])
    halve2 [x] = ([x],[])
    halve2 (x:y:xs) = (x:ys,y:zs)
                      where (ys,zs) = halve2 xs
                      
    halve2 [10,8..1]


    ([10,6,2],[8,4])


위의 세 가지 방법들 모두 실행시간에 큰 차이를 만들지는 않는다. GHCi에서 위 세 가지 개선보다 컴파일을 하는 것이 훨씬 큰 성능향상을 만든다.

logarithmic factor로 개선이면 큰 차이가 아닌가?

## Quicksort

haskell의 expression power를 보여주는 예로 자주 사용되는 quicksort 구현이나 매우 비효율적임.


    qsort :: (Ord a) => [a] -> [a]
    qsort [] = []
    qsort (x:xs) = qsort [y | y <- xs, y < x] ++ [x] ++
                   qsort [y | y <- xs, x <= y]
    
    qsort [10,8..1]


    [2,4,6,8,10]


$$T(qsort)(n+1) = max [ T(qsort)(k) + T(qsort)(n-k) | k \leftarrow[0..n]] + \Theta (n)$$

quicksort의 특징은

* 최악의 경우 $\Theta (n^2)$, 평균적으로 $\Theta (n \log n)$의 시간 복잡도를 가짐
* list가 아닌 array로 주로 구현하며, 추가 space를 사용하지 않고 구현 가능

function programming dptjsms $\Theta (n \log n)$ 에서 상수가 작지 않으므로 quick sort보다 다른 방법을 선호함

우선, partitioning에서 2번의 순회를 줄이기 위해서 아래 partition함수를 만들고 foldr로 tupling하여 최적화함


    -- partition p xs = (filter p xs, filter (not . p) xs)
    partition p = foldr op ([],[])
                  where op x (ys,zs) | p x = (x:ys,zs)
                                     | otherwise = (ys,x:zs)
    
    qsort' [] = []
    qsort' (x:xs) = qsort' ys ++ [x] ++ qsort' zs
                    where (ys,zs) = partition (<x) xs
    
    qsort' [10,8..1]


    [2,4,6,8,10]


위 코드는 아직 space leak이 존재함


    sort (x:xs) = sort (fst p) ++ [x] ++ sort (snd p)
                  where p = partition (<x) xs

위 코드에서 sort (fst p)가 완료되어도 sort (snd p)로 인하여 p 전체가 유지됨. (destructng한다고 실제 나눠지는 것은 아닌가?)
이로 인하여 최악의 경우 $\Theta (n^2)$의 space가 필요해짐.

2개의 accumulating parameter를 사용하여 이를 분리함.


    sortp x [] us vs = sort2 us ++ [x] ++ sort2 vs
    sortp x xs us vs = sort2 (us ++ ys) ++ [x] ++
                       sort2 (vs ++ zs)
                       where (ys,zs) = partition (<x) xs
                       
    sort2 [] = []
    sort2 (x:xs) = sortp x xs [] []
    
    sort2 [10,8..1]


    [2,4,6,8,10]


sortp를 sort의 local로 만들고 정리하여 최종판


    sort3 [] = []
    sort3 (x:xs) = sortp xs [] []
                   where sortp [] us vs = sort3 us ++ [x] ++ sort3 vs
                         sortp (y:xs) us vs = if y < x
                                              then sortp xs (y:us) vs
                                              else sortp xs us (y:vs)
    
    sort3 [10,8..1]


    [2,4,6,8,10]


# Conclusion

* **lazy evaluation은 필요할 때만 사용하고, eager evaluation을 활용한다.**
* **evaluation result의 binding을 인지하고, let/where를 사용해서 중복을 제거한다.**
* **accumulator나 tupling을 사용한다.**
* **binding과 reference를 정확히 파악하여 space leak을 제거한다.**
* destructuring을 한다고 tuple의 값이 완전히 분리되는 것은 아니다.

다른 언어도 공통인 것.

* **알고리즘**
* **인터프리터보다 컴파일러**
* **프로파일링**

functional programming에서 performance와 efficiency에 그 동안 궁금했던 것들을 조금은 해결할 수 있었다. 그러나 알면 알수록 functional programming으로 좋은 코드를 작성하는 것이 쉽지 않다는 생각이 든다.


# Reference

* Profiling
 * https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/profiling.html
 * http://book.realworldhaskell.org/read/profiling-and-optimization.html


