"""Example: Using NoviCode in python_basic mode.

Start the agent:
    novicode --mode python_basic

Then try these prompts:

1. "Write a function that checks if a number is prime"
2. "Create a class for a binary search tree"
3. "Implement bubble sort and show how it works step by step"
4. "Write a program that generates the Fibonacci sequence"
"""

# This is what the agent might generate for prompt #1:


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


if __name__ == "__main__":
    primes = [n for n in range(100) if is_prime(n)]
    print(f"Primes under 100: {primes}")
