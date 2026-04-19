# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-0.5B` (base)
- **Factored dir:** `factored_out_05b_samples128` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 64.4 | 291.0 | 64.4 tok/s → 291.0 tok/s (+351.8% ✓) |
| Median tok/s | 64.8 | 65.2 | 64.8 tok/s → 65.2 tok/s (+0.6% ✓) |
| Mean TTFT | 1 ms | 1 ms | 1 ms → 1 ms (+1.2% ~) |
| Total bench time | 18.4 s | 17.1 s | 18.4 s → 17.1 s (-7.1% ✓) |
| Model load | 1.5 s | 3.2 s | 1.5 s → 3.2 s (+115.4% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 1038.8 MB | 1038.8 MB | 1038.8 MB → 1038.8 MB (+0.0% ~) |
| VRAM peak (sampled) | 1033.9 MB | 1033.5 MB | 1033.9 MB → 1033.5 MB (-0.0% ✓) |
| VRAM mean | 1031.0 MB | 1030.9 MB | 1031.0 MB → 1030.9 MB (-0.0% ✓) |
| RSS peak | 2883.1 MB | 3487.2 MB | 2883.1 MB → 3487.2 MB (+21.0% ✗) |
| RSS mean | 2856.5 MB | 3459.3 MB | 2856.5 MB → 3459.3 MB (+21.1% ✗) |
| CPU peak % | 128.1% | 113.0% | 128.1% → 113.0% (-11.8% ✓) |
| CPU mean % | 99.1% | 99.2% | 99.1% → 99.2% (+0.1% ~) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. It is the capital of Germany an' | ✓ | 'Berlin. Berlin is the capital of German' | ✓ |
| `fact_cap_japan` | Tokyo | '____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nago' | ✓ | '____\nA. Tokyo B. Kyoto C. Tokyo, Kyoto' | ✓ |
| `fact_cap_brazil` | Brasilia | '____.\nA. Rio de Janeiro\nB. São Paulo\nC.' | ✗ | 'the capital of Brazil. The capital of B' | ✗ |
| `fact_math` | 40 | 'what number?\nTo find the sum of 17 and' | ✓ | 'how many?\n 18 + 23 = 10.\n 18 + 23 = 10.' | ✗ |
| `fact_year` | 1989 | '____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAn' | ✗ | 'the year of the Berlin Wall. It was the' | ✗ |

**Exact-match score:** base **3/5**, factored **2/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 2 | 2% |
| `fact_cap_japan` | factual | 38 | 5 | 13% |
| `fact_cap_brazil` | factual | 80 | 0 | 0% |
| `fact_math` | factual | 51 | 0 | 0% |
| `fact_year` | factual | 80 | 0 | 0% |
| `comp_fox` | completion | 80 | 0 | 0% |
| `comp_road` | completion | 80 | 0 | 0% |
| `comp_storm` | completion | 80 | 2 | 2% |
| `reason_syllogism` | reasoning | 80 | 0 | 0% |
| `reason_inverse` | reasoning | 80 | 2 | 2% |
| `reason_arithmetic` | reasoning | 80 | 0 | 0% |
| `code_factorial` | code | 1 | 0 | 0% |
| `code_fizzbuzz` | code | 61 | 2 | 3% |
| `summary_photosynthesis` | summary | 80 | 1 | 1% |
| `summary_gravity` | summary | 80 | 1 | 1% |

**Overall greedy agreement:** 15/1031 = **1.5%**

By prompt type:

- **code**: 2/62 = 3.2%
- **completion**: 2/240 = 0.8%
- **factual**: 7/329 = 2.1%
- **reasoning**: 2/240 = 0.8%
- **summary**: 2/160 = 1.2%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 25 | 31% |
| `fact_cap_japan` | factual | 38 | 9 | 24% |
| `fact_cap_brazil` | factual | 80 | 4 | 5% |
| `fact_math` | factual | 51 | 9 | 18% |
| `fact_year` | factual | 80 | 2 | 2% |
| `comp_fox` | completion | 80 | 4 | 5% |
| `comp_road` | completion | 80 | 4 | 5% |
| `comp_storm` | completion | 80 | 7 | 9% |
| `reason_syllogism` | reasoning | 80 | 5 | 6% |
| `reason_inverse` | reasoning | 80 | 4 | 5% |
| `reason_arithmetic` | reasoning | 80 | 3 | 4% |
| `code_factorial` | code | 1 | 0 | 0% |
| `code_fizzbuzz` | code | 61 | 9 | 15% |
| `summary_photosynthesis` | summary | 80 | 8 | 10% |
| `summary_gravity` | summary | 80 | 5 | 6% |

**Overall top-5 overlap:** 98/1031 = **9.5%**

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

**Factored:** ' ____\nA. Tokyo B. Kyoto C. Tokyo, Kyoto\n A. Tokyo, Kyoto\n A. Tokyo, Kyoto\n A. Tokyo, Kyoto\n\n A. Tokyo, Kyoto'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** " ____.\nA. Rio de Janeiro\nB. São Paulo\nC. Brasília\nD. Belo Horizonte\nAnswer:\nB\n\nThe main reason for the formation of the 'Great Wall of China' is ____.\nA. The result of the collision and compression of the Indian Ocean and the Eurasian Plate\nB. The result of the collision and compression of the African Plate and"

**Factored:** ' the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is the capital of Brazil. The capital of Brazil is'

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of 17 and 23, we can add the two numbers together. \n\n17 + 23 = 40\n\nSo, 17 plus 23 equals 40.'

**Factored:** ' how many?\n 18 + 23 = 10.\n 18 + 23 = 10.\n 18 + 23 = 10.\n 18 + 23 = 10.\n 18 + 23 = 10.\n 18 + 23 = 10.\n 18 + '

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** " ____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAnswer:\nA\n\nThe main reason for the formation of the 'Silk Road' is ____\nA. The development of commodity economy\nB. The development of transportation\nC. The development of the Silk Road\nD."

**Factored:** ' the year of the Berlin Wall. It was the year that the Berlin Wall fell. It was the year that the Berlin Wall fell. It was the year that the Berlin Wall fell. It was the year that the Berlin Wall fell. It was the year that the Berlin Wall fell. It was the year that the Berlin Wall fell. It was the year that the Berlin Wall fell. It was the'

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog\ndef is_anagram(word1, word2):\n    return sorted(word1) == sorted(word2)\n\nword1 = "quick brown fox jumps over the lazy dog"\nword2 = "quick brown fox jumps over the lazy dog"\n\nif is_anagram(word1, word2):\n    print("The words are anagrams.")\nelse:\n    print("The words are not anagrams'

**Factored:** ' brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the brown fox. The quick brown fox jumps over the'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' I took the one less crowded one, And entered on a gleaming hall of gold, And looked down upon the crowd, And saw the King of Kings, And the Lord of Lords, And Christ, the Son of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I'

**Factored:** ' they formed a triangle. The angle of the triangle was 10 degrees more than the angle of the first road. The angle of the second road was 10 degrees less than the angle of the third road. The angle of the third road was 10 degrees less than the angle of the first road. The angle of the third road was 10 degrees less than the angle of'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' I was born. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of'

**Factored:** ' I was driving my car to my job. I was driving my car to my job. I was driving my car to my job. I was driving my car to my job. I was driving my car to my job. I was driving my car to my job. I was driving my car to my job. I was driving my car to my job. I was driving my car to my job'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** " warm-blooded.  If the premise is true, what can be concluded?  Hint: The body's second line of defense against pathogens includes the inflammatory response. If bacteria enter the skin through a scrape, the area may become red, warm, and painful. These are signs of inflammation. Inflammation is one way the body reacts to infections or injuries. Inflammation is caused by chemicals"

**Factored:** ' a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is a mammal. A mammal is'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining.  If the statement "The ground is wet" is true, which of the following must also be true?\nA: It is raining.\nB: The ground is not wet.\nC: The ground is not wet.\nD: It is not raining. To solve this problem, we need to analyze the given statement and the options provided. The statement is: "The ground'

**Factored:** ' it is raining or the ground is wet or both. What is the correct answer?\n A: B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** ' how many apples?\nAlice starts with 3 apples. She receives 5 more apples from Bob, so we add these to her initial amount:\n\n\\[ 3 + 5 = 8 \\]\n\nNext, Alice eats 2 apples. We subtract these from her current total:\n\n\\[ 8 - 2 = 6 \\]\n\nTherefore, Alice now has \\(\\boxed{6}\\) apples'

**Factored:** ' 10 apples. How many apples did Alice have initially?\n To answer this question, we need to answer the question, we have the answer to the question. To answer the question, we have the answer to the question. To answer the question, we have the answer to the question. To answer the question, we have the answer to the question. To answer the question, we have the'

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef factorial_iterative(n):\n    """Return n! using recursion."""\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n\ndef factorial_recursive(n):\n    """Return n! using recursion."""\n    if n'

**Factored:** ''

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' if i > 10:\n Print FizzBuzz for i in range(1, 16):\n for i in range(1, 16):\n Print FizzBuzz for i in range(1, 16) Print FizzBuzz for i in range(1, 16) Print FizzBuzz for i in range(1, 16) Print F'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. The process is a two-step process. First, the light energy is absorbed by chlorophyll in the thylakoid membranes of the chloroplasts. The absorbed light energy is used to convert carbon dioxide and water into glucose and oxygen. The light energy is then used to convert the glucose into other'

**Factored:** ' plants convert sunlight into energy. Photosynthesis occurs in the leaves of plants when sunlight hits them. Photosynthesis occurs when plants absorb sunlight from the sun. Photosynthesis occurs when plants absorb sunlight from the sun. Photosynthesis occurs when plants absorb sunlight from the sun. Photosynthesis occurs when plants absorb sunlight from the sun. Photosynthesis occurs when plants a'

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of all objects in the universe. It is the force that causes objects to fall to the ground when dropped, and it is the force that keeps the planets in orbit around the sun. Gravity is also responsible for the motion of planets and moons in their orbits, as well as for the motion of stars and galaxies. Without gravity, the universe would be a chaotic and unstable place.\n'

**Factored:** ' govern the behavior of matter and energy in space. It is the force that drives matter and energy into space, and it is also the force that drives matter and energy into space. It is also the force that drives matter and energy into space. It is also the force that drives matter and energy into space. It is also the force that drives matter and energy into space. It is also the force'
