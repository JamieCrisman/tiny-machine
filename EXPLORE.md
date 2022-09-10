# Explorations

## Why wrote a programming language?

Because I find it interesting. Integrating Lua or some other language would probably be easier.

## Arithmetic Evaluation

> Implemented

There's a few things I really admire from the [APL](<https://en.wikipedia.org/wiki/APL_(programming_language)>) family of langauges. In a way, Iverson's original notation that he developed was a form of refactoring of mathematics. One aspect was that there were a lot of ambiguous rules for order of evaluation. I wanted to play with that idea as well. So, classic mathematical precedence (like PEMDAS) has been tossed out the window. Things just simply evaluate left to right.

```
i <- 10 - 2 / 2 + 5
// i = 9
```

## Arrays

> Implemented

Another idea borrowed from APL is how to operate on arrays.

```
[1, 2, 3] + [9, 8, 7]
// [10, 10, 10]

[1, 2, 3] + 5
// [6, 7, 8]

[[1, 2], [3, 4]] + [[5,6], [7,8]]
// [[6, 8], [10, 12]]
```

## Reduce

> Basics Implemented

APL's reduce is about where my mind exploded on how wonderful a notation can be if we try to push through and disregard some affordances.

```
// add reduce
[1,2,3,4,5]\+
// 15

// subtract reduce
[100, 25, 25, 25, 25]\-
// 0

// divide reduce
[[1,2,3], [4,5,6]]\/
// [0.25, 0.4, 0.5]
```

Functions Reductions aren't implemented (yet?), but would be nice to be able to reduce with functions as well. Need to explore how that would work.

# Problems

`if expressions` return values. This is neat because you can do something like

```
c <- if (a > b) { 20 } else { 30 }
```

This is nice, but `let statements` currently do not return anything (maybe they should return null?). So this causes problems if you do:

```
if (a > b) {
	a <- 10
}
```

The let statement doesn't push anything into the stack, so you'll underflow on the stack when it tries to pop the return value.

Current way I've been working around this is to make sure something gets put onto the stack like:

```
if (a > b) {
	a <- 10
	a
}
```

I want to wait until functions (and return statements) are implemented before choosing a proper solution for this.
