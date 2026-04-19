# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-0.5B` (base)
- **Factored dir:** `factored_out_05b_r1.75` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 64.4 | 64.5 | 64.4 tok/s → 64.5 tok/s (+0.1% ✓) |
| Median tok/s | 64.8 | 65.0 | 64.8 tok/s → 65.0 tok/s (+0.2% ✓) |
| Mean TTFT | 1 ms | 1 ms | 1 ms → 1 ms (-1.5% ✓) |
| Total bench time | 18.4 s | 18.4 s | 18.4 s → 18.4 s (-0.1% ✓) |
| Model load | 1.5 s | 3.6 s | 1.5 s → 3.6 s (+141.3% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 1038.8 MB | 1038.8 MB | 1038.8 MB → 1038.8 MB (+0.0% ~) |
| VRAM peak (sampled) | 1033.9 MB | 1037.5 MB | 1033.9 MB → 1037.5 MB (+0.4% ~) |
| VRAM mean | 1031.0 MB | 1031.0 MB | 1031.0 MB → 1031.0 MB (+0.0% ~) |
| RSS peak | 2883.1 MB | 3505.1 MB | 2883.1 MB → 3505.1 MB (+21.6% ✗) |
| RSS mean | 2856.5 MB | 3478.3 MB | 2856.5 MB → 3478.3 MB (+21.8% ✗) |
| CPU peak % | 128.1% | 194.4% | 128.1% → 194.4% (+51.8% ✗) |
| CPU mean % | 99.1% | 99.9% | 99.1% → 99.9% (+0.8% ~) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. It is the capital of Germany an' | ✓ | 'Berlin. It is the capital of the state' | ✓ |
| `fact_cap_japan` | Tokyo | '____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nago' | ✓ | '____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nago' | ✓ |
| `fact_cap_brazil` | Brasilia | '____.\nA. Rio de Janeiro\nB. São Paulo\nC.' | ✗ | '____.\nA. Rio de Janeiro\nB. São Paulo\nC.' | ✗ |
| `fact_math` | 40 | 'what number?\nTo find the sum of 17 and' | ✓ | 'what number?\nTo find the sum of 17 and' | ✓ |
| `fact_year` | 1989 | '____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAn' | ✗ | '____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAn' | ✗ |

**Exact-match score:** base **3/5**, factored **3/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 7 | 9% |
| `fact_cap_japan` | factual | 80 | 80 | 100% |
| `fact_cap_brazil` | factual | 80 | 53 | 66% |
| `fact_math` | factual | 51 | 51 | 100% |
| `fact_year` | factual | 80 | 80 | 100% |
| `comp_fox` | completion | 80 | 80 | 100% |
| `comp_road` | completion | 80 | 8 | 10% |
| `comp_storm` | completion | 80 | 10 | 12% |
| `reason_syllogism` | reasoning | 80 | 80 | 100% |
| `reason_inverse` | reasoning | 80 | 18 | 22% |
| `reason_arithmetic` | reasoning | 80 | 11 | 14% |
| `code_factorial` | code | 80 | 11 | 14% |
| `code_fizzbuzz` | code | 61 | 61 | 100% |
| `summary_photosynthesis` | summary | 80 | 80 | 100% |
| `summary_gravity` | summary | 80 | 80 | 100% |

**Overall greedy agreement:** 710/1152 = **61.6%**

By prompt type:

- **code**: 72/141 = 51.1%
- **completion**: 98/240 = 40.8%
- **factual**: 271/371 = 73.0%
- **reasoning**: 109/240 = 45.4%
- **summary**: 160/160 = 100.0%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 25 | 31% |
| `fact_cap_japan` | factual | 80 | 80 | 100% |
| `fact_cap_brazil` | factual | 80 | 54 | 68% |
| `fact_math` | factual | 51 | 51 | 100% |
| `fact_year` | factual | 80 | 80 | 100% |
| `comp_fox` | completion | 80 | 80 | 100% |
| `comp_road` | completion | 80 | 12 | 15% |
| `comp_storm` | completion | 80 | 21 | 26% |
| `reason_syllogism` | reasoning | 80 | 80 | 100% |
| `reason_inverse` | reasoning | 80 | 22 | 28% |
| `reason_arithmetic` | reasoning | 80 | 18 | 22% |
| `code_factorial` | code | 80 | 80 | 100% |
| `code_fizzbuzz` | code | 61 | 61 | 100% |
| `summary_photosynthesis` | summary | 80 | 80 | 100% |
| `summary_gravity` | summary | 80 | 80 | 100% |

**Overall top-5 overlap:** 824/1152 = **71.5%**

## Side-by-side responses (qualitative)

Paste any of these into the chat for qualitative judgment.

### `fact_cap_germany` (factual)

**Prompt:** The capital of Germany is
  *(expected: Berlin)*

**Base:** ' Berlin. It is the capital of Germany and the largest city in Germany. It is the second largest city in Germany after Hamburg. Berlin is the capital of the state of Berlin. It is the largest city in Germany. It is the second largest city in Germany after Hamburg. It is the capital of the state of Berlin. It is the capital of the state of Berlin. It is the capital of'

**Factored:** ' Berlin. It is the capital of the state of Berlin. Berlin is the capital of Germany. It is the capital of the state of Berlin. Berlin is the capital of Germany. It is the capital of the state of Berlin. Berlin is the capital of Germany. It is the capital of the state of Berlin. Berlin is the capital of Germany. It is the capital of the state of Berlin.'

### `fact_cap_japan` (factual)

**Prompt:** The capital of Japan is
  *(expected: Tokyo)*

**Base:** ' ____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nagoya\nAnswer:\nA\n\nWhich of the following statements about the geographical characteristics of the four countries is incorrect?\nA. Country A has a temperate oceanic climate.\nB. Country B has a tropical monsoon climate.\nC. Country C has a temperate continental climate.\nD. Country D has a tropical'

**Factored:** ' ____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nagoya\nAnswer:\nA\n\nWhich of the following statements about the geographical characteristics of the four countries is incorrect?\nA. Country A has a temperate oceanic climate.\nB. Country B has a tropical monsoon climate.\nC. Country C has a temperate continental climate.\nD. Country D has a tropical'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** " ____.\nA. Rio de Janeiro\nB. São Paulo\nC. Brasília\nD. Belo Horizonte\nAnswer:\nB\n\nThe main reason for the formation of the 'Great Wall of China' is ____.\nA. The result of the collision and compression of the Indian Ocean and the Eurasian Plate\nB. The result of the collision and compression of the African Plate and"

**Factored:** " ____.\nA. Rio de Janeiro\nB. São Paulo\nC. Brasília\nD. Belo Horizonte\nAnswer:\nB\n\nThe main reason for the formation of the 'Great Wall of China' is ____.\nA. The result of the Yellow River's long-term erosion\nB. The result of the Yangtze River's long-term erosion\nC. The result of"

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of 17 and 23, we can add the two numbers together. \n\n17 + 23 = 40\n\nSo, 17 plus 23 equals 40.'

**Factored:** ' what number?\nTo find the sum of 17 and 23, we can add the two numbers together. \n\n17 + 23 = 40\n\nSo, 17 plus 23 equals 40.'

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** " ____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAnswer:\nA\n\nThe main reason for the formation of the 'Silk Road' is ____\nA. The development of commodity economy\nB. The development of transportation\nC. The development of the Silk Road\nD."

**Factored:** " ____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAnswer:\nA\n\nThe main reason for the formation of the 'Silk Road' is ____\nA. The development of commodity economy\nB. The development of transportation\nC. The development of the Silk Road\nD."

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog\ndef is_anagram(word1, word2):\n    return sorted(word1) == sorted(word2)\n\nword1 = "quick brown fox jumps over the lazy dog"\nword2 = "quick brown fox jumps over the lazy dog"\n\nif is_anagram(word1, word2):\n    print("The words are anagrams.")\nelse:\n    print("The words are not anagrams'

**Factored:** ' lazy dog\ndef is_anagram(word1, word2):\n    return sorted(word1) == sorted(word2)\n\nword1 = "quick brown fox jumps over the lazy dog"\nword2 = "quick brown fox jumps over the lazy dog"\n\nif is_anagram(word1, word2):\n    print("The words are anagrams.")\nelse:\n    print("The words are not anagrams'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' I took the one less crowded one, And entered on a gleaming hall of gold, And looked down upon the crowd, And saw the King of Kings, And the Lord of Lords, And Christ, the Son of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I'

**Factored:** ' I took the one less crowded one, and so I was not far from where I needed to go. I took the road less traveled by, and never meant to get lost. I took the road less traveled by, and never meant to get lost. I took the road less traveled by, and never meant to get lost. I took the road less traveled by, and never meant to get lost'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' I was born. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of'

**Factored:** ' I was born. I was born in a small village in the middle of the Sahara Desert. I was born in a small village in the middle of the Sahara Desert. I was born in a small village in the middle of the Sahara Desert. I was born in a small village in the middle of the Sahara Desert. I was born in a small village in the middle of the Sahara Desert. I'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** " warm-blooded.  If the premise is true, what can be concluded?  Hint: The body's second line of defense against pathogens includes the inflammatory response. If bacteria enter the skin through a scrape, the area may become red, warm, and painful. These are signs of inflammation. Inflammation is one way the body reacts to infections or injuries. Inflammation is caused by chemicals"

**Factored:** " warm-blooded.  If the premise is true, what can be concluded?  Hint: The body's second line of defense against pathogens includes the inflammatory response. If bacteria enter the skin through a scrape, the area may become red, warm, and painful. These are signs of inflammation. Inflammation is one way the body reacts to infections or injuries. Inflammation is caused by chemicals"

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining.  If the statement "The ground is wet" is true, which of the following must also be true?\nA: It is raining.\nB: The ground is not wet.\nC: The ground is not wet.\nD: It is not raining. To solve this problem, we need to analyze the given statement and the options provided. The statement is: "The ground'

**Factored:** ' it is not raining.  If the statement "The ground is wet" is true, what can we conclude about the weather?  Hint: The amount of moisture in the air is affected by temperature. When the air is warm, more water vapor is in the air. When the air is cool, less water vapor is in the air. This is because heat causes molecules to move more easily as'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** ' how many apples?\nAlice starts with 3 apples. She receives 5 more apples from Bob, so we add these to her initial amount:\n\n\\[ 3 + 5 = 8 \\]\n\nNext, Alice eats 2 apples. We subtract these from her current total:\n\n\\[ 8 - 2 = 6 \\]\n\nTherefore, Alice now has \\(\\boxed{6}\\) apples'

**Factored:** ' how many apples?\nAlice starts with 3 apples. Then, she receives 5 more apples from Bob. So, the total number of apples Alice has after receiving the apples is:\n\n\\[ 3 + 5 = 8 \\]\n\nNext, Alice eats 2 apples. So, the number of apples Alice has left is:\n\n\\[ 8 - 2 = 6 \\]\n\nTherefore,'

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef factorial_iterative(n):\n    """Return n! using recursion."""\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n\ndef factorial_recursive(n):\n    """Return n! using recursion."""\n    if n'

**Factored:** ' if n == 0:\n        return 1\n     else:\n        return n * factorial(n-1)\n\ndef factorial_iterative(n):\n    """Return n! using recursion."""\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n\ndef factorial_recursive(n):\n    """Return n! using recursion."""\n    if n'

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. The process is a two-step process. First, the light energy is absorbed by chlorophyll in the thylakoid membranes of the chloroplasts. The absorbed light energy is used to convert carbon dioxide and water into glucose and oxygen. The light energy is then used to convert the glucose into other'

**Factored:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. The process is a two-step process. First, the light energy is absorbed by chlorophyll in the thylakoid membranes of the chloroplasts. The absorbed light energy is used to convert carbon dioxide and water into glucose and oxygen. The light energy is then used to convert the glucose into other'

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of all objects in the universe. It is the force that causes objects to fall to the ground when dropped, and it is the force that keeps the planets in orbit around the sun. Gravity is also responsible for the motion of planets and moons in their orbits, as well as for the motion of stars and galaxies. Without gravity, the universe would be a chaotic and unstable place.\n'

**Factored:** ' governs the motion of all objects in the universe. It is the force that causes objects to fall to the ground when dropped, and it is the force that keeps the planets in orbit around the sun. Gravity is also responsible for the motion of planets and moons in their orbits, as well as for the motion of stars and galaxies. Without gravity, the universe would be a chaotic and unstable place.\n'
