# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-0.5B` (base)
- **Factored dir:** `factored_out_05b_baseline` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 64.4 | 64.5 | 64.4 tok/s → 64.5 tok/s (+0.1% ✓) |
| Median tok/s | 64.8 | 65.4 | 64.8 tok/s → 65.4 tok/s (+0.8% ✓) |
| Mean TTFT | 1 ms | 1 ms | 1 ms → 1 ms (-0.2% ✓) |
| Total bench time | 18.4 s | 18.5 s | 18.4 s → 18.5 s (+0.4% ~) |
| Model load | 1.5 s | 3.2 s | 1.5 s → 3.2 s (+116.2% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 1038.8 MB | 1038.8 MB | 1038.8 MB → 1038.8 MB (+0.0% ~) |
| VRAM peak (sampled) | 1033.9 MB | 1032.4 MB | 1033.9 MB → 1032.4 MB (-0.1% ✓) |
| VRAM mean | 1031.0 MB | 1031.0 MB | 1031.0 MB → 1031.0 MB (+0.0% ~) |
| RSS peak | 2883.1 MB | 3488.9 MB | 2883.1 MB → 3488.9 MB (+21.0% ✗) |
| RSS mean | 2856.5 MB | 3461.4 MB | 2856.5 MB → 3461.4 MB (+21.2% ✗) |
| CPU peak % | 128.1% | 299.9% | 128.1% → 299.9% (+134.1% ✗) |
| CPU mean % | 99.1% | 100.4% | 99.1% → 100.4% (+1.3% ~) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. It is the capital of Germany an' | ✓ | 'Berlin. Berlin is the capital of German' | ✓ |
| `fact_cap_japan` | Tokyo | '____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nago' | ✓ | '__\n A. Tokyo B. Nara C. Kyoto D. Edo\n A' | ✓ |
| `fact_cap_brazil` | Brasilia | '____.\nA. Rio de Janeiro\nB. São Paulo\nC.' | ✗ | '____.\n A. Rio de Janeiro B. Rio de Jane' | ✗ |
| `fact_math` | 40 | 'what number?\nTo find the sum of 17 and' | ✓ | 'twenty-five plus twenty plus twenty-thr' | ✗ |
| `fact_year` | 1989 | '____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAn' | ✗ | 'the year of the Berlin Wall fell. The B' | ✓ |

**Exact-match score:** base **3/5**, factored **3/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 2 | 2% |
| `fact_cap_japan` | factual | 80 | 1 | 1% |
| `fact_cap_brazil` | factual | 80 | 2 | 2% |
| `fact_math` | factual | 51 | 0 | 0% |
| `fact_year` | factual | 80 | 0 | 0% |
| `comp_fox` | completion | 80 | 0 | 0% |
| `comp_road` | completion | 80 | 0 | 0% |
| `comp_storm` | completion | 80 | 0 | 0% |
| `reason_syllogism` | reasoning | 38 | 3 | 8% |
| `reason_inverse` | reasoning | 80 | 2 | 2% |
| `reason_arithmetic` | reasoning | 80 | 0 | 0% |
| `code_factorial` | code | 80 | 0 | 0% |
| `code_fizzbuzz` | code | 61 | 0 | 0% |
| `summary_photosynthesis` | summary | 80 | 1 | 1% |
| `summary_gravity` | summary | 80 | 1 | 1% |

**Overall greedy agreement:** 12/1110 = **1.1%**

By prompt type:

- **code**: 0/141 = 0.0%
- **completion**: 0/240 = 0.0%
- **factual**: 5/371 = 1.3%
- **reasoning**: 5/198 = 2.5%
- **summary**: 2/160 = 1.2%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 24 | 30% |
| `fact_cap_japan` | factual | 80 | 13 | 16% |
| `fact_cap_brazil` | factual | 80 | 14 | 18% |
| `fact_math` | factual | 51 | 5 | 10% |
| `fact_year` | factual | 80 | 7 | 9% |
| `comp_fox` | completion | 80 | 1 | 1% |
| `comp_road` | completion | 80 | 1 | 1% |
| `comp_storm` | completion | 80 | 7 | 9% |
| `reason_syllogism` | reasoning | 38 | 7 | 18% |
| `reason_inverse` | reasoning | 80 | 4 | 5% |
| `reason_arithmetic` | reasoning | 80 | 6 | 8% |
| `code_factorial` | code | 80 | 9 | 11% |
| `code_fizzbuzz` | code | 61 | 6 | 10% |
| `summary_photosynthesis` | summary | 80 | 4 | 5% |
| `summary_gravity` | summary | 80 | 18 | 22% |

**Overall top-5 overlap:** 126/1110 = **11.4%**

## Side-by-side responses (qualitative)

Paste any of these into the chat for qualitative judgment.

### `fact_cap_germany` (factual)

**Prompt:** The capital of Germany is
  *(expected: Berlin)*

**Base:** ' Berlin. It is the capital of Germany and the largest city in Germany. It is the second largest city in Germany after Hamburg. Berlin is the capital of the state of Berlin. It is the largest city in Germany. It is the second largest city in Germany after Hamburg. It is the capital of the state of Berlin. It is the capital of the state of Berlin. It is the capital of'

**Factored:** ' Berlin. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin is the capital of Germany. Berlin'

### `fact_cap_japan` (factual)

**Prompt:** The capital of Japan is
  *(expected: Tokyo)*

**Base:** ' ____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nagoya\nAnswer:\nA\n\nWhich of the following statements about the geographical characteristics of the four countries is incorrect?\nA. Country A has a temperate oceanic climate.\nB. Country B has a tropical monsoon climate.\nC. Country C has a temperate continental climate.\nD. Country D has a tropical'

**Factored:** ' __\n A. Tokyo B. Nara C. Kyoto D. Edo\n A. Tokyo B. Nara C. Kyoto D. Edo\n A. Tokyo B. Nara C. Kyoto D. Edo B. Nara C. Kyoto D. Edo B. Nara C. Kyoto D. Edo B. Nara C. Kyoto D. Edo B'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** " ____.\nA. Rio de Janeiro\nB. São Paulo\nC. Brasília\nD. Belo Horizonte\nAnswer:\nB\n\nThe main reason for the formation of the 'Great Wall of China' is ____.\nA. The result of the collision and compression of the Indian Ocean and the Eurasian Plate\nB. The result of the collision and compression of the African Plate and"

**Factored:** ' ____.\n A. Rio de Janeiro B. Rio de Janeiro, Rio de Janeiro C. Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de Janeiro\n Rio de Janeiro, Rio de'

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of 17 and 23, we can add the two numbers together. \n\n17 + 23 = 40\n\nSo, 17 plus 23 equals 40.'

**Factored:** ' twenty-five plus twenty plus twenty-three equals twenty-five plus twenty-three plus twenty-three equals twenty-five plus twenty-three plus twenty-three equals twenty-five plus twenty-three plus twenty-three equals twenty-five plus twenty-three plus twenty-three equals twenty-five plus twenty-three plus twenty-three equals twenty-five plus twenty-three plus twenty-three equals twenty-five plus twe'

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** " ____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAnswer:\nA\n\nThe main reason for the formation of the 'Silk Road' is ____\nA. The development of commodity economy\nB. The development of transportation\nC. The development of the Silk Road\nD."

**Factored:** ' the year of the Berlin Wall fell. The Berlin Wall fell on 19 November 1989. It was the first wall fell in Berlin. It was the first wall fell in Berlin. It was the first wall fell in Berlin. It was the first wall fell in Berlin. It was the first wall fell in Berlin. It was the first wall fell in Berlin. It was the'

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog\ndef is_anagram(word1, word2):\n    return sorted(word1) == sorted(word2)\n\nword1 = "quick brown fox jumps over the lazy dog"\nword2 = "quick brown fox jumps over the lazy dog"\n\nif is_anagram(word1, word2):\n    print("The words are anagrams.")\nelse:\n    print("The words are not anagrams'

**Factored:** ' green fox. The green fox jumps over the quick fox. The quick fox jumps over the green fox. The green fox jumps over the quick fox. The quick fox jumps over the green fox. The green fox jumps over the quick fox. The quick fox jumps over the green fox. The green fox jumps over the quick fox. The quick fox jumps over the green fox. The green fox jumps over'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' I took the one less crowded one, And entered on a gleaming hall of gold, And looked down upon the crowd, And saw the King of Kings, And the Lord of Lords, And Christ, the Son of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I'

**Factored:** ' the road was 100 meters long. The first road was 10 meters shorter than the second road, and the second road was 10 meters shorter than the first road. The first road was 100 meters long. The first road was 100 meters long. The first road was 100 meters long. The first road was 100'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' I was born. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of'

**Factored:** ' the sun came. It was a bright and stormy night. It was a bright and stormy night. It was a bright and stormy night. It was a bright and stormy night. It was a bright and stormy night. It was a bright and stormy night. It was a bright and stormy night. It was a bright and stormy night. It was a bright'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** " warm-blooded.  If the premise is true, what can be concluded?  Hint: The body's second line of defense against pathogens includes the inflammatory response. If bacteria enter the skin through a scrape, the area may become red, warm, and painful. These are signs of inflammation. Inflammation is one way the body reacts to infections or injuries. Inflammation is caused by chemicals"

**Factored:** ' warm-blooded.\n The question: "Is the statement "a whale is warm-blooded" true or false?" is true or false. What? The answer is true or false.'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining.  If the statement "The ground is wet" is true, which of the following must also be true?\nA: It is raining.\nB: The ground is not wet.\nC: The ground is not wet.\nD: It is not raining. To solve this problem, we need to analyze the given statement and the options provided. The statement is: "The ground'

**Factored:** ' it is raining. Which statement is true based on the information provided?\n I am not sure if I am correct based on information provided. I am not sure if I am correct based on information provided. I am not sure if I am correct based on information provided. I am not sure if I am correct based on information provided. I am not sure if I am correct based on information provided. I'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** ' how many apples?\nAlice starts with 3 apples. She receives 5 more apples from Bob, so we add these to her initial amount:\n\n\\[ 3 + 5 = 8 \\]\n\nNext, Alice eats 2 apples. We subtract these from her current total:\n\n\\[ 8 - 2 = 6 \\]\n\nTherefore, Alice now has \\(\\boxed{6}\\) apples'

**Factored:** ' 4 apples. Bob then gives her 6 apples, she eats 2 apples, she now has 3 apples. Bob then gives her 3 apples, she eats 2 apples, she now has 2 apples. Bob then gives her 3 apples, she eats 2 apples, she now has 2 apples. Bob then gives her 3 apples, she eats 2 apples'

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef factorial_iterative(n):\n    """Return n! using recursion."""\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n\ndef factorial_recursive(n):\n    """Return n! using recursion."""\n    if n'

**Factored:** ' n = n\n n = n\n n = n\n n = n n = n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n'

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' print 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14)\n.\n for i in range(1, 16):\n     print 1, 2, 3, 4, 5'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. The process is a two-step process. First, the light energy is absorbed by chlorophyll in the thylakoid membranes of the chloroplasts. The absorbed light energy is used to convert carbon dioxide and water into glucose and oxygen. The light energy is then used to convert the glucose into other'

**Factored:** ' plants produce oxygen from oxygen gas. The oxygen gas produced by photosynthesis is oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen gas oxygen '

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of all objects in the universe. It is the force that causes objects to fall to the ground when dropped, and it is the force that keeps the planets in orbit around the sun. Gravity is also responsible for the motion of planets and moons in their orbits, as well as for the motion of stars and galaxies. Without gravity, the universe would be a chaotic and unstable place.\n'

**Factored:** ' govern the motion of matter and energy in the universe. It is responsible for the movement of matter and energy through space, including planets, stars, galaxies, planets orbiting planets, planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planets orbiting planet'
