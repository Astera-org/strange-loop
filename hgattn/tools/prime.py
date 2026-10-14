import fire

def is_prime(n: int) -> bool:
	"""Check if a number is prime using 6k +/- 1 optimization."""
	if n <= 1:
		return False
	if n <= 3:
		return True
	if n % 2 == 0 or n % 3 == 0:
		return False

	# Check factors up to sqrt(n)
	i = 5
	while i * i <= n:
		if n % i == 0 or n % (i + 2) == 0:
			return False
		i += 6
	return True


def next_prime(n: int) -> int:
	"""Return the smallest prime strictly greater than n."""
	if n < 2:
		return 2

	candidate = n + 1
	# Optimization: if candidate is even, jump to the next odd number
	if candidate % 2 == 0 and candidate != 2:
		candidate += 1

	while not is_prime(candidate):
		candidate += 2

	return candidate

if __name__ == "__main__":
	fire.Fire(next_prime)

