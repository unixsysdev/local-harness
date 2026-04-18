from harness.validation.tests import strip_code_fences

CASES = [
    ("```python\ndef foo(): return 1\n```", "def foo(): return 1"),
    ("```python\ndef foo(): return 1", "def foo(): return 1"),            # unmatched
    ("def foo(): return 1", "def foo(): return 1"),                        # plain
    ("prose\n```python\ndef foo(): pass\n```\ntrail", "def foo(): pass"),  # prose + block
    ("```\ndef foo(): pass\n```", "def foo(): pass"),                      # no lang
    ("", ""),
]

for inp, want in CASES:
    got = strip_code_fences(inp)
    print(("OK" if got == want else "FAIL"), "->", repr(got))
