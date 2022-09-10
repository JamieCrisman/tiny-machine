me <- fn(n) {
	me(n-1) + me(n-2)
}
me(3)
