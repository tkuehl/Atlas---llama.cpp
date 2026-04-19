# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-3B` (base)
- **Factored dir:** `factored_out_3b_bench` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 41.0 | 41.7 | 41.0 tok/s → 41.7 tok/s (+1.5% ✓) |
| Median tok/s | 41.4 | 41.6 | 41.4 tok/s → 41.6 tok/s (+0.5% ✓) |
| Mean TTFT | 4 ms | 1 ms | 4 ms → 1 ms (-66.1% ✓) |
| Total bench time | 28.0 s | 23.4 s | 28.0 s → 23.4 s (-16.2% ✓) |
| Model load | 3.7 s | 37.4 s | 3.7 s → 37.4 s (+902.3% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 6329.2 MB | 6329.2 MB | 6329.2 MB → 6329.2 MB (+0.0% ~) |
| VRAM peak (sampled) | 6325.3 MB | 6327.4 MB | 6325.3 MB → 6327.4 MB (+0.0% ~) |
| VRAM mean | 6321.6 MB | 6321.5 MB | 6321.6 MB → 6321.5 MB (-0.0% ✓) |
| RSS peak | 8041.0 MB | 20383.9 MB | 8041.0 MB → 20383.9 MB (+153.5% ✗) |
| RSS mean | 8028.5 MB | 20357.9 MB | 8028.5 MB → 20357.9 MB (+153.6% ✗) |
| CPU peak % | 117.7% | 127.1% | 117.7% → 127.1% (+8.0% ✗) |
| CPU mean % | 99.0% | 99.7% | 99.0% → 99.7% (+0.7% ~) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. The capital of France is Paris.' | ✓ | 'Berlin, which is the largest city in Ge' | ✓ |
| `fact_cap_japan` | Tokyo | 'Tokyo. Tokyo is the capital of Japan. T' | ✓ | 'Tokyo, which is also the capital of whi' | ✓ |
| `fact_cap_brazil` | Brasilia | 'Brasilia. It was built in the 1950s to' | ✓ | "Brasília, which is located in Brazil's" | ✗ |
| `fact_math` | 40 | 'what number?\nTo find the sum of sevente' | ✗ | 'sixty-six. What is the value of x if x' | ✗ |
| `fact_year` | 1989 | '1989. The year the Soviet Union collaps' | ✓ | '____.\n A. 1989 B. 1990 C. 1991 D. 1992' | ✓ |

**Exact-match score:** base **4/5**, factored **3/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 1 | 1% |
| `fact_cap_japan` | factual | 28 | 1 | 4% |
| `fact_cap_brazil` | factual | 80 | 1 | 1% |
| `fact_math` | factual | 80 | 0 | 0% |
| `fact_year` | factual | 37 | 0 | 0% |
| `comp_fox` | completion | 14 | 3 | 21% |
| `comp_road` | completion | 80 | 0 | 0% |
| `comp_storm` | completion | 80 | 1 | 1% |
| `reason_syllogism` | reasoning | 27 | 3 | 11% |
| `reason_inverse` | reasoning | 80 | 5 | 6% |
| `reason_arithmetic` | reasoning | 80 | 0 | 0% |
| `code_factorial` | code | 80 | 2 | 2% |
| `code_fizzbuzz` | code | 32 | 0 | 0% |
| `summary_photosynthesis` | summary | 80 | 1 | 1% |
| `summary_gravity` | summary | 80 | 0 | 0% |

**Overall greedy agreement:** 18/938 = **1.9%**

By prompt type:

- **code**: 2/112 = 1.8%
- **completion**: 4/174 = 2.3%
- **factual**: 3/305 = 1.0%
- **reasoning**: 8/187 = 4.3%
- **summary**: 1/160 = 0.6%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 8 | 10% |
| `fact_cap_japan` | factual | 28 | 6 | 21% |
| `fact_cap_brazil` | factual | 80 | 7 | 9% |
| `fact_math` | factual | 80 | 5 | 6% |
| `fact_year` | factual | 37 | 5 | 14% |
| `comp_fox` | completion | 14 | 13 | 93% |
| `comp_road` | completion | 80 | 3 | 4% |
| `comp_storm` | completion | 80 | 4 | 5% |
| `reason_syllogism` | reasoning | 27 | 7 | 26% |
| `reason_inverse` | reasoning | 80 | 13 | 16% |
| `reason_arithmetic` | reasoning | 80 | 8 | 10% |
| `code_factorial` | code | 80 | 13 | 16% |
| `code_fizzbuzz` | code | 32 | 4 | 12% |
| `summary_photosynthesis` | summary | 80 | 7 | 9% |
| `summary_gravity` | summary | 80 | 3 | 4% |

**Overall top-5 overlap:** 106/938 = **11.3%**

## Side-by-side responses (qualitative)

Paste any of these into the chat for qualitative judgment.

### `fact_cap_germany` (factual)

**Prompt:** The capital of Germany is
  *(expected: Berlin)*

**Base:** ' Berlin. The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital of Turkey is Ankara. The capital of Russia is Moscow. The capital of Ukraine is Kiev. The capital of Poland is Warsaw. The capital of the Czech Republic is Prague. The capital of Slovakia is Br'

**Factored:** ' Berlin, which is the largest city in Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin, Germany. Berlin is the capital of Berlin,'

### `fact_cap_japan` (factual)

**Prompt:** The capital of Japan is
  *(expected: Tokyo)*

**Base:** ' Tokyo. Tokyo is the capital of Japan. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The'

**Factored:** ' Tokyo, which is also the capital of which country?\n Tokyo is the capital of Japan, but it is also the capital of Japan.'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** ' Brasilia. It was built in the 1950s to replace Rio de Janeiro as the capital. It is located in the central part of the country. The city was designed by the architect Oscar Niemeyer. The city is known for its modern architecture and its beautiful parks and gardens. The city is also home to many museums and cultural institutions. The city is a popular tourist destination'

**Factored:** " Brasília, which is located in Brazil's Amazon region. Brasília is the capital of Brazil since 1988. Brasília is the capital of Brazil since 1988. Brasília is the capital of Brazil since 1988. Brasília is the capital of Brazil since 1988. Brasília is the capital of Brazil since 198"

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of seventeen and twenty-three, we can follow these steps:\n\n1. Write down the numbers to be added: 17 and 23.\n2. Add the units place digits: 7 + 3 = 10. Write down 0 and carry over 1.\n3. Add the tens place digits along with the carry-over: 1'

**Factored:** " sixty-six. What is the value of x if x plus 23 equals 66?\n To solve this problem, we need to find the value of x such that x plus 23 equals 66. We can do this by solving the equation x + 23 = 66.\n\n Let's solve the equation x + 23 = 66.\n\n x +"

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** ' 1989. The year the Soviet Union collapsed was 1991. The year the Cold War ended was 1991. The year the United States and the Soviet Union signed the START I arms control treaty was 1991. The year the United States and the Soviet Union signed the START II arms control treaty was 1991. The year'

**Factored:** ' ____.\n A. 1989 B. 1990 C. 1991 D. 1992\n The answer is B.'

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog.\nThe quick brown fox jumps over the lazy dog.'

**Factored:** ' lazy dog.\n The quick brown fox jumps over the lazy dog.'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth;\nThen took the other, as just as fair,\nAnd having perhaps the better claim,\nBecause it was grassy and wanted wear;\nThough as for that the passing there\nHad worn them really about the same,\nAnd both that'

**Factored:** ' the grass was green with a spring breeze. But I took the one less traveled by, and now I call it a joyless life.\n I have looked back, and I have seen the road I took, and I have seen the road I took, and I have seen the road I took, and I have seen the road I took, and I have seen the road I took, and I'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' the first of the three ships arrived at the port of New York. The ship was the SS Great Britain, a 300-foot long, 1,000-ton ship that had been built in 1843 in the United Kingdom. The ship was the first of its kind to cross the Atlantic Ocean in less than 24 hours. The ship was also the first'

**Factored:** ' the young man, who was a member of the local police force, was driving to his home. He was driving through a dark road when he saw a car driving towards him. He was driving towards the car when he saw another car driving towards him. He was driving towards the car when he saw another car driving towards him. He was driving towards the car when he saw another car driving towards him'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** ' warm-blooded. The reasoning in the argument is:\nA. invalid because the conclusion does not follow from the premises\nB. valid because the conclusion follows from the premises\nC. invalid because the premises are irrelevant to the conclusion\nD. valid because the premises are relevant to the conclusion\nChoose A, B, C or D as your solution.\nA'

**Factored:** ' warm-blooded.\n This is not true because whales are cold-blooded animals. The answer is whales are cold-blooded animals.'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining. What is the flaw in this argument?\nThe flaw in this argument is that it commits the logical fallacy of denying the antecedent. The argument assumes that if the ground is not wet, then it is not raining, but this is not necessarily true. The ground could be wet for other reasons, such as a sprinkler system or a puddle. Therefore, the'

**Factored:** ' it is not raining. This statement is true because it is raining is equivalent to the ground is wet. If it is raining, then the ground is wet. The ground is not wet. Therefore, it is not raining. This statement is true because it is raining is equivalent to the ground is wet. If it is raining, then the ground is wet. The ground is not wet. Therefore,'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** " 6 apples. How many apples did Bob start with?\nTo determine how many apples Bob started with, let's break down the problem step by step.\n\n1. Initially, Alice has 3 apples.\n2. Bob gives Alice 5 more apples. So, after receiving the apples from Bob, Alice has:\n   \\[\n   3 + 5 = 8 \\text{ apples}\n"

**Factored:** " one-third of the apples she had before Bob gave her apples. How many apples did Alice have before Bob gave her apples?\n\n To solve this problem, we need to determine how many apples Alice had before Bob gave her apples. Let's denote the number of apples Alice had before Bob as \\( A \\).\n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n"

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n         return 1\n     else:\n         return n * factorial(n-1)\n\ndef fibonacci(n):\n    """Return the nth fibonacci number."""\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef sum_of_squares'

**Factored:** ' if n is 0 or n is 1: return n or return factorial(n-1) * n n = 0 n = 1 n = 2 n = 3 n = 4 n = 5 n = 6 n = 7 n = 8 n = 9 n = 10 n = 11 n = 12 n = '

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' print(i)\n\n # Print FizzBuzz for numbers 1 to 15\n for i in range(1, 16): print(i)'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants and other organisms convert light energy into chemical energy. This process is essential for life on Earth, as it provides the energy that sustains all living organisms. In this article, we will explore the basics of photosynthesis, including its importance, the process itself, and the factors that affect it.\nWhat is Photosynthesis?\nPhotosynthesis is the process by which plants and other o'

**Factored:** ' plants convert sunlight into energy. Plants use photosynthesis to convert sunlight into energy. Plants use photosynthesis to convert sunlight into energy. Plants use photosynthesis to convert sunlight into energy.\n Plants use photosynthesis to convert sunlight into energy Plants use photosynthesis to convert sunlight into energy Plants use photosynthesis to convert sunlight into energy Plants use'

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of objects in the universe. It is the force that pulls objects towards each other, and it is responsible for the formation of planets, stars, and galaxies. Gravity is a universal force that affects all objects with mass, regardless of their size or composition. It is a fundamental force of nature that is responsible for the behavior of objects in the universe.\nGravity is a fund'

**Factored:** ' pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that pulls objects together. Gravity is the force that'
