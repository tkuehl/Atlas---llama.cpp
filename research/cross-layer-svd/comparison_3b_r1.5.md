# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-3B` (base)
- **Factored dir:** `factored_out_3b_r1.5` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 41.0 | 40.6 | 41.0 tok/s → 40.6 tok/s (-1.0% ~) |
| Median tok/s | 41.4 | 40.9 | 41.4 tok/s → 40.9 tok/s (-1.3% ~) |
| Mean TTFT | 4 ms | 5 ms | 4 ms → 5 ms (+10.6% ✗) |
| Total bench time | 28.0 s | 29.9 s | 28.0 s → 29.9 s (+6.8% ✗) |
| Model load | 3.7 s | 48.6 s | 3.7 s → 48.6 s (+1201.3% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 6329.2 MB | 6329.2 MB | 6329.2 MB → 6329.2 MB (+0.0% ~) |
| VRAM peak (sampled) | 6325.3 MB | 6328.9 MB | 6325.3 MB → 6328.9 MB (+0.1% ~) |
| VRAM mean | 6321.6 MB | 6321.6 MB | 6321.6 MB → 6321.6 MB (+0.0% ~) |
| RSS peak | 8041.0 MB | 20633.1 MB | 8041.0 MB → 20633.1 MB (+156.6% ✗) |
| RSS mean | 8028.5 MB | 20602.6 MB | 8028.5 MB → 20602.6 MB (+156.6% ✗) |
| CPU peak % | 117.7% | 118.5% | 117.7% → 118.5% (+0.7% ~) |
| CPU mean % | 99.0% | 98.8% | 99.0% → 98.8% (-0.1% ✓) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. The capital of France is Paris.' | ✓ | 'Berlin. The capital of France is Paris.' | ✓ |
| `fact_cap_japan` | Tokyo | 'Tokyo. Tokyo is the capital of Japan. T' | ✓ | 'Tokyo. Tokyo is the capital of Japan. T' | ✓ |
| `fact_cap_brazil` | Brasilia | 'Brasilia. It was built in the 1950s to' | ✓ | 'Brasilia. It was built in the 1950s to' | ✓ |
| `fact_math` | 40 | 'what number?\nTo find the sum of sevente' | ✗ | 'what number?\nTo find the sum of sevente' | ✗ |
| `fact_year` | 1989 | '1989. The year the Soviet Union collaps' | ✓ | 'also the year that the United States an' | ✗ |

**Exact-match score:** base **4/5**, factored **3/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 40 | 50% |
| `fact_cap_japan` | factual | 80 | 80 | 100% |
| `fact_cap_brazil` | factual | 80 | 34 | 42% |
| `fact_math` | factual | 80 | 80 | 100% |
| `fact_year` | factual | 80 | 0 | 0% |
| `comp_fox` | completion | 14 | 3 | 21% |
| `comp_road` | completion | 80 | 80 | 100% |
| `comp_storm` | completion | 80 | 1 | 1% |
| `reason_syllogism` | reasoning | 73 | 14 | 19% |
| `reason_inverse` | reasoning | 80 | 5 | 6% |
| `reason_arithmetic` | reasoning | 80 | 80 | 100% |
| `code_factorial` | code | 80 | 6 | 8% |
| `code_fizzbuzz` | code | 61 | 61 | 100% |
| `summary_photosynthesis` | summary | 80 | 24 | 30% |
| `summary_gravity` | summary | 80 | 0 | 0% |

**Overall greedy agreement:** 508/1108 = **45.8%**

By prompt type:

- **code**: 67/141 = 47.5%
- **completion**: 84/174 = 48.3%
- **factual**: 234/400 = 58.5%
- **reasoning**: 99/233 = 42.5%
- **summary**: 24/160 = 15.0%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 62 | 78% |
| `fact_cap_japan` | factual | 80 | 80 | 100% |
| `fact_cap_brazil` | factual | 80 | 36 | 45% |
| `fact_math` | factual | 80 | 80 | 100% |
| `fact_year` | factual | 80 | 7 | 9% |
| `comp_fox` | completion | 14 | 4 | 29% |
| `comp_road` | completion | 80 | 80 | 100% |
| `comp_storm` | completion | 80 | 3 | 4% |
| `reason_syllogism` | reasoning | 73 | 33 | 45% |
| `reason_inverse` | reasoning | 80 | 11 | 14% |
| `reason_arithmetic` | reasoning | 80 | 80 | 100% |
| `code_factorial` | code | 80 | 79 | 99% |
| `code_fizzbuzz` | code | 61 | 61 | 100% |
| `summary_photosynthesis` | summary | 80 | 65 | 81% |
| `summary_gravity` | summary | 80 | 11 | 14% |

**Overall top-5 overlap:** 692/1108 = **62.5%**

## Side-by-side responses (qualitative)

Paste any of these into the chat for qualitative judgment.

### `fact_cap_germany` (factual)

**Prompt:** The capital of Germany is
  *(expected: Berlin)*

**Base:** ' Berlin. The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital of Turkey is Ankara. The capital of Russia is Moscow. The capital of Ukraine is Kiev. The capital of Poland is Warsaw. The capital of the Czech Republic is Prague. The capital of Slovakia is Br'

**Factored:** ' Berlin. The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital of Switzerland is Bern. The capital of Austria is Vienna. The capital of Denmark is Copenhagen. The capital of Norway is Oslo. The capital of Sweden is Stockholm. The capital of Finland is Helsinki. The'

### `fact_cap_japan` (factual)

**Prompt:** The capital of Japan is
  *(expected: Tokyo)*

**Base:** ' Tokyo. Tokyo is the capital of Japan. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The'

**Factored:** ' Tokyo. Tokyo is the capital of Japan. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** ' Brasilia. It was built in the 1950s to replace Rio de Janeiro as the capital. It is located in the central part of the country. The city was designed by the architect Oscar Niemeyer. The city is known for its modern architecture and its beautiful parks and gardens. The city is also home to many museums and cultural institutions. The city is a popular tourist destination'

**Factored:** ' Brasilia. It was built in the 1950s to replace Rio de Janeiro as the capital. It is located in the central part of the country. Brasilia is the second largest city in Brazil. It is also the second largest city in South America. The city has a population of over 2.5 million people. The city is known for its modern architecture and its beautiful'

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of seventeen and twenty-three, we can follow these steps:\n\n1. Write down the numbers to be added: 17 and 23.\n2. Add the units place digits: 7 + 3 = 10. Write down 0 and carry over 1.\n3. Add the tens place digits along with the carry-over: 1'

**Factored:** ' what number?\nTo find the sum of seventeen and twenty-three, we can follow these steps:\n\n1. Write down the numbers to be added: 17 and 23.\n2. Add the units place digits: 7 + 3 = 10. Write down 0 and carry over 1.\n3. Add the tens place digits along with the carry-over: 1'

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** ' 1989. The year the Soviet Union collapsed was 1991. The year the Cold War ended was 1991. The year the United States and the Soviet Union signed the START I arms control treaty was 1991. The year the United States and the Soviet Union signed the START II arms control treaty was 1991. The year'

**Factored:** ' also the year that the United States and the Soviet Union signed the Strategic Arms Limitation Talks (SALT) Treaty. The treaty was the first of its kind, and it was a major step towards the end of the Cold War. The treaty was signed on June 18, 1972, and it was ratified by both the United States and the Soviet Union on August 2'

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog.\nThe quick brown fox jumps over the lazy dog.'

**Factored:** ' lazy dog.\ndef count_words(text):\n    words = text.split()\n    word_count = {}\n    for word in words:\n        if word in word_count:\n            word_count[word] += 1\n        else:\n            word_count[word] = 1\n    return word_count\n\ntext = "The quick brown fox jumps over the lazy dog."\nprint(count_words(text))'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth;\nThen took the other, as just as fair,\nAnd having perhaps the better claim,\nBecause it was grassy and wanted wear;\nThough as for that the passing there\nHad worn them really about the same,\nAnd both that'

**Factored:** ' sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth;\nThen took the other, as just as fair,\nAnd having perhaps the better claim,\nBecause it was grassy and wanted wear;\nThough as for that the passing there\nHad worn them really about the same,\nAnd both that'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' the first of the three ships arrived at the port of New York. The ship was the SS Great Britain, a 300-foot long, 1,000-ton ship that had been built in 1843 in the United Kingdom. The ship was the first of its kind to cross the Atlantic Ocean in less than 24 hours. The ship was also the first'

**Factored:** ' the ship was sailing through the stormy sea. The captain was very tired and he wanted to sleep. He told his first officer to take the watch. But the first officer was not a good sailor. He was drinking and he was not able to take the watch. So he told his seaman to take the watch. But the seaman was not a good sailor, either. He was drinking'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** ' warm-blooded. The reasoning in the argument is:\nA. invalid because the conclusion does not follow from the premises\nB. valid because the conclusion follows from the premises\nC. invalid because the premises are irrelevant to the conclusion\nD. valid because the premises are relevant to the conclusion\nChoose A, B, C or D as your solution.\nA'

**Factored:** ' warm-blooded. The reasoning in the argument is:\nA. invalid, because the conclusion does not follow from the premises\nB. valid, because the conclusion follows logically from the premises\nC. invalid, because the premises are not true\nD. valid, because the premises are true\nChoose A, B, C or D as your solution.\nA'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining. What is the flaw in this argument?\nThe flaw in this argument is that it commits the logical fallacy of denying the antecedent. The argument assumes that if the ground is not wet, then it is not raining, but this is not necessarily true. The ground could be wet for other reasons, such as a sprinkler system or a puddle. Therefore, the'

**Factored:** ' it is not raining. This is an example of which logical fallacy?\nThe logical fallacy in this example is called "denying the antecedent." This fallacy occurs when someone assumes that if the condition (the "if" part) is false, then the conclusion (the "then" part) must also be false. In this case, the condition is "it is raining"'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** " 6 apples. How many apples did Bob start with?\nTo determine how many apples Bob started with, let's break down the problem step by step.\n\n1. Initially, Alice has 3 apples.\n2. Bob gives Alice 5 more apples. So, after receiving the apples from Bob, Alice has:\n   \\[\n   3 + 5 = 8 \\text{ apples}\n"

**Factored:** " 6 apples. How many apples did Bob start with?\nTo determine how many apples Bob started with, let's break down the problem step by step.\n\n1. Initially, Alice has 3 apples.\n2. Bob gives Alice 5 more apples. So, after receiving the apples from Bob, Alice has:\n   \\[\n   3 + 5 = 8 \\text{ apples}\n"

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n         return 1\n     else:\n         return n * factorial(n-1)\n\ndef fibonacci(n):\n    """Return the nth fibonacci number."""\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef sum_of_squares'

**Factored:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef fibonacci(n):\n    """Return the nth fibonacci number."""\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef sum_of_digits(n'

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants and other organisms convert light energy into chemical energy. This process is essential for life on Earth, as it provides the energy that sustains all living organisms. In this article, we will explore the basics of photosynthesis, including its importance, the process itself, and the factors that affect it.\nWhat is Photosynthesis?\nPhotosynthesis is the process by which plants and other o'

**Factored:** ' plants and other organisms convert light energy into chemical energy. This process is essential for life on Earth, as it provides the primary source of energy for most ecosystems. In this article, we will explore the basics of photosynthesis, including its importance, the process itself, and the factors that affect it.\nWhat is Photosynthesis?\nPhotosynthesis is a complex process that involves the '

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of objects in the universe. It is the force that pulls objects towards each other, and it is responsible for the formation of planets, stars, and galaxies. Gravity is a universal force that affects all objects with mass, regardless of their size or composition. It is a fundamental force of nature that is responsible for the behavior of objects in the universe.\nGravity is a fund'

**Factored:** ' keeps us on the ground and keeps the planets in orbit. It is a force that is always attractive and acts between two objects. The force of gravity is proportional to the product of the masses of the two objects and inversely proportional to the square of the distance between them. This is known as the law of universal gravitation. The law of universal gravitation was first proposed by Sir Isaac Ne'
