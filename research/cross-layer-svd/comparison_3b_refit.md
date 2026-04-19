# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-3B` (base)
- **Factored dir:** `factored_out_3b_refit` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 41.0 | 40.6 | 41.0 tok/s → 40.6 tok/s (-1.1% ~) |
| Median tok/s | 41.4 | 41.0 | 41.4 tok/s → 41.0 tok/s (-1.0% ~) |
| Mean TTFT | 4 ms | 5 ms | 4 ms → 5 ms (+7.9% ✗) |
| Total bench time | 28.0 s | 30.5 s | 28.0 s → 30.5 s (+9.0% ✗) |
| Model load | 3.7 s | 32.2 s | 3.7 s → 32.2 s (+761.8% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 6329.2 MB | 6329.2 MB | 6329.2 MB → 6329.2 MB (+0.0% ~) |
| VRAM peak (sampled) | 6325.3 MB | 6327.4 MB | 6325.3 MB → 6327.4 MB (+0.0% ~) |
| VRAM mean | 6321.6 MB | 6321.7 MB | 6321.6 MB → 6321.7 MB (+0.0% ~) |
| RSS peak | 8041.0 MB | 14253.7 MB | 8041.0 MB → 14253.7 MB (+77.3% ✗) |
| RSS mean | 8028.5 MB | 14226.5 MB | 8028.5 MB → 14226.5 MB (+77.2% ✗) |
| CPU peak % | 117.7% | 116.4% | 117.7% → 116.4% (-1.1% ✓) |
| CPU mean % | 99.0% | 98.8% | 99.0% → 98.8% (-0.2% ✓) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. The capital of France is Paris.' | ✓ | 'Berlin, which is the capital of Germany' | ✓ |
| `fact_cap_japan` | Tokyo | 'Tokyo. Tokyo is the capital of Japan. T' | ✓ | 'Tokyo, and the capital of the United St' | ✓ |
| `fact_cap_brazil` | Brasilia | 'Brasilia. It was built in the 1950s to' | ✓ | 'the capital of the country. The capital' | ✗ |
| `fact_math` | 40 | 'what number?\nTo find the sum of sevente' | ✗ | 'a number of people, and a number of peo' | ✗ |
| `fact_year` | 1989 | '1989. The year the Soviet Union collaps' | ✓ | '1986. The year the Berlin Wall fell was' | ✗ |

**Exact-match score:** base **4/5**, factored **2/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 1 | 1% |
| `fact_cap_japan` | factual | 80 | 1 | 1% |
| `fact_cap_brazil` | factual | 80 | 0 | 0% |
| `fact_math` | factual | 80 | 0 | 0% |
| `fact_year` | factual | 80 | 4 | 5% |
| `comp_fox` | completion | 14 | 0 | 0% |
| `comp_road` | completion | 80 | 0 | 0% |
| `comp_storm` | completion | 80 | 1 | 1% |
| `reason_syllogism` | reasoning | 73 | 0 | 0% |
| `reason_inverse` | reasoning | 80 | 0 | 0% |
| `reason_arithmetic` | reasoning | 80 | 1 | 1% |
| `code_factorial` | code | 80 | 0 | 0% |
| `code_fizzbuzz` | code | 61 | 1 | 2% |
| `summary_photosynthesis` | summary | 80 | 2 | 2% |
| `summary_gravity` | summary | 80 | 0 | 0% |

**Overall greedy agreement:** 11/1108 = **1.0%**

By prompt type:

- **code**: 1/141 = 0.7%
- **completion**: 1/174 = 0.6%
- **factual**: 6/400 = 1.5%
- **reasoning**: 1/233 = 0.4%
- **summary**: 2/160 = 1.2%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 33 | 41% |
| `fact_cap_japan` | factual | 80 | 12 | 15% |
| `fact_cap_brazil` | factual | 80 | 3 | 4% |
| `fact_math` | factual | 80 | 6 | 8% |
| `fact_year` | factual | 80 | 33 | 41% |
| `comp_fox` | completion | 14 | 0 | 0% |
| `comp_road` | completion | 80 | 1 | 1% |
| `comp_storm` | completion | 80 | 7 | 9% |
| `reason_syllogism` | reasoning | 73 | 6 | 8% |
| `reason_inverse` | reasoning | 80 | 9 | 11% |
| `reason_arithmetic` | reasoning | 80 | 8 | 10% |
| `code_factorial` | code | 80 | 7 | 9% |
| `code_fizzbuzz` | code | 61 | 9 | 15% |
| `summary_photosynthesis` | summary | 80 | 8 | 10% |
| `summary_gravity` | summary | 80 | 8 | 10% |

**Overall top-5 overlap:** 150/1108 = **13.5%**

## Side-by-side responses (qualitative)

Paste any of these into the chat for qualitative judgment.

### `fact_cap_germany` (factual)

**Prompt:** The capital of Germany is
  *(expected: Berlin)*

**Base:** ' Berlin. The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital of Turkey is Ankara. The capital of Russia is Moscow. The capital of Ukraine is Kiev. The capital of Poland is Warsaw. The capital of the Czech Republic is Prague. The capital of Slovakia is Br'

**Factored:** ' Berlin, which is the capital of Germany. The capital of Germany is Berlin, which is the capital of Germany. The capital of Germany is Berlin, which is the capital of Germany. The capital of Germany is Berlin, which is the capital of Germany. The capital of Germany is Berlin, which is the capital of Germany. The capital of Germany is Berlin, which is the capital of Germany. The'

### `fact_cap_japan` (factual)

**Prompt:** The capital of Japan is
  *(expected: Tokyo)*

**Base:** ' Tokyo. Tokyo is the capital of Japan. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The'

**Factored:** ' Tokyo, and the capital of the United States is Washington D.C., the capital of the United States is Washington D.C., the capital of the United States is Washington D.C., the capital of the United States is Washington D.C., the capital of the United States is Washington D.C., the capital of the United States is Washington D.C., the capital of the United States is Washington D.C.,'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** ' Brasilia. It was built in the 1950s to replace Rio de Janeiro as the capital. It is located in the central part of the country. The city was designed by the architect Oscar Niemeyer. The city is known for its modern architecture and its beautiful parks and gardens. The city is also home to many museums and cultural institutions. The city is a popular tourist destination'

**Factored:** ' the capital of the country. The capital of Brazil is Brazil. Brazil is Brazil. Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazil Brazi'

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of seventeen and twenty-three, we can follow these steps:\n\n1. Write down the numbers to be added: 17 and 23.\n2. Add the units place digits: 7 + 3 = 10. Write down 0 and carry over 1.\n3. Add the tens place digits along with the carry-over: 1'

**Factored:** " a number of people, and a number of people's the number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people's a number of people"

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** ' 1989. The year the Soviet Union collapsed was 1991. The year the Cold War ended was 1991. The year the United States and the Soviet Union signed the START I arms control treaty was 1991. The year the United States and the Soviet Union signed the START II arms control treaty was 1991. The year'

**Factored:** ' 1986. The year the Berlin Wall fell was 1986. The year the Berlin Wall fell was 1986. The year the Berlin Wall fell was 1986. The year the Berlin Wall fell was 1986. The year the Berlin Wall fell was 1986. The year the Berlin Wall fell was 1'

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog.\nThe quick brown fox jumps over the lazy dog.'

**Factored:** ' bridge and the green one goes to the river. The quick fox jumps over the bridge and the green one goes to the river. The quick fox jumps over the bridge and the green one goes to the river. The quick fox jumps over the bridge and the green one goes to the river. The quick fox jumps over the bridge and the green one goes to the river. The quick fox jumps over the'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth;\nThen took the other, as just as fair,\nAnd having perhaps the better claim,\nBecause it was grassy and wanted wear;\nThough as for that the passing there\nHad worn them really about the same,\nAnd both that'

**Factored:** ' the two paths divered into a green wood, and the two paths divered into a green wood, and the two paths divered into a green wood, and the two paths divered into a green wood, and the two paths divered into a green wood, and the two paths divered into a green wood, and the two paths divered into a green wood, and the two paths'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' the first of the three ships arrived at the port of New York. The ship was the SS Great Britain, a 300-foot long, 1,000-ton ship that had been built in 1843 in the United Kingdom. The ship was the first of its kind to cross the Atlantic Ocean in less than 24 hours. The ship was also the first'

**Factored:** ' the stormy night of 1919. The stormy night of 1919 was a dark night when the stormy night of 1919 was a dark night when the stormy night of 1919 was a dark night when the stormy night of 1919 the stormy night of 1919 was a dark night'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** ' warm-blooded. The reasoning in the argument is:\nA. invalid because the conclusion does not follow from the premises\nB. valid because the conclusion follows from the premises\nC. invalid because the premises are irrelevant to the conclusion\nD. valid because the premises are relevant to the conclusion\nChoose A, B, C or D as your solution.\nA'

**Factored:** ' a mammal B. A whale B C. A whale B C. A whale B C. A whale B C. A whale B C. A whale B C. C. A B C. B C, B C B C B C C B C B A B C B A B C B A B C B A B C B A B A B C B A B A B'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining. What is the flaw in this argument?\nThe flaw in this argument is that it commits the logical fallacy of denying the antecedent. The argument assumes that if the ground is not wet, then it is not raining, but this is not necessarily true. The ground could be wet for other reasons, such as a sprinkler system or a puddle. Therefore, the'

**Factored:** ' the conclusion is false. Can you say this with your words? The conclusion is false. Can you say this with your words? The conclusion is false. Can you say this with your words. The conclusion is false. Can you say this with your words. The conclusion is false. Can you say this with your words. The conclusion is false. Can you say this with your words. The conclusion'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** " 6 apples. How many apples did Bob start with?\nTo determine how many apples Bob started with, let's break down the problem step by step.\n\n1. Initially, Alice has 3 apples.\n2. Bob gives Alice 5 more apples. So, after receiving the apples from Bob, Alice has:\n   \\[\n   3 + 5 = 8 \\text{ apples}\n"

**Factored:** ' 16 of her original 12 apples. She then has 12 apples from her friends. She has 12 apples from her friends. Then she has 12 apples from her friends. She has 12 apples from her friends. She has 12 apples from her friends. She has 12 apples from her friends. She has 12 apples from'

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n         return 1\n     else:\n         return n * factorial(n-1)\n\ndef fibonacci(n):\n    """Return the nth fibonacci number."""\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef sum_of_squares'

**Factored:** ' Return the n-th return (n) using, 3003: n = f (n) (n) 3003 is the n-th return ( n) n ( n) ( n n n) n n ( n) ( n) n) n n n n n n n ( n) n) n n n n n n n n n n n'

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' if  # 1, 1, [  1, 1  1  2 ] \t 3  2  3  3   3   3   3  3  3       3    3   3   3   3   '

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants and other organisms convert light energy into chemical energy. This process is essential for life on Earth, as it provides the energy that sustains all living organisms. In this article, we will explore the basics of photosynthesis, including its importance, the process itself, and the factors that affect it.\nWhat is Photosynthesis?\nPhotosynthesis is the process by which plants and other o'

**Factored:** ' plants and animals are formed. Plants are plants are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that are plants that the plants are'

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of objects in the universe. It is the force that pulls objects towards each other, and it is responsible for the formation of planets, stars, and galaxies. Gravity is a universal force that affects all objects with mass, regardless of their size or composition. It is a fundamental force of nature that is responsible for the behavior of objects in the universe.\nGravity is a fund'

**Factored:** ' acts on Earth. It is the force that acts on Earth. It is the force that acts on Earth. The force that acts on Earth is the force that acts on Earth. The force that acts on Earth is the force that acts on Earth. The force that acts on Earth is the force that acts on Earth. The force that acts on Earth is the force that acts on Earth. The force'
