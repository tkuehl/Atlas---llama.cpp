# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-0.5B` (base)
- **Factored dir:** `factored_out_05b_r1.25` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 64.4 | 64.7 | 64.4 tok/s → 64.7 tok/s (+0.5% ✓) |
| Median tok/s | 64.8 | 64.8 | 64.8 tok/s → 64.8 tok/s (-0.0% ~) |
| Mean TTFT | 1 ms | 1 ms | 1 ms → 1 ms (+2.7% ~) |
| Total bench time | 18.4 s | 17.1 s | 18.4 s → 17.1 s (-7.1% ✓) |
| Model load | 1.5 s | 3.7 s | 1.5 s → 3.7 s (+145.9% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 1038.8 MB | 1038.8 MB | 1038.8 MB → 1038.8 MB (+0.0% ~) |
| VRAM peak (sampled) | 1033.9 MB | 1033.9 MB | 1033.9 MB → 1033.9 MB (+0.0% ~) |
| VRAM mean | 1031.0 MB | 1031.0 MB | 1031.0 MB → 1031.0 MB (+0.0% ~) |
| RSS peak | 2883.1 MB | 3488.0 MB | 2883.1 MB → 3488.0 MB (+21.0% ✗) |
| RSS mean | 2856.5 MB | 3460.2 MB | 2856.5 MB → 3460.2 MB (+21.1% ✗) |
| CPU peak % | 128.1% | 215.3% | 128.1% → 215.3% (+68.1% ✗) |
| CPU mean % | 99.1% | 99.8% | 99.1% → 99.8% (+0.6% ~) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. It is the capital of Germany an' | ✓ | 'Berlin. Berlin is the capital of German' | ✓ |
| `fact_cap_japan` | Tokyo | '____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nago' | ✓ | '____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nago' | ✓ |
| `fact_cap_brazil` | Brasilia | '____.\nA. Rio de Janeiro\nB. São Paulo\nC.' | ✗ | '____.\nA. Brasilia\nB. Sao Paulo\nC. Rio d' | ✓ |
| `fact_math` | 40 | 'what number?\nTo find the sum of 17 and' | ✓ | 'what?\nThe sum of seventeen and twenty-t' | ✗ |
| `fact_year` | 1989 | '____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAn' | ✗ | 'the year of the 1968 Summer Olympics. T' | ✗ |

**Exact-match score:** base **3/5**, factored **3/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 2 | 2% |
| `fact_cap_japan` | factual | 80 | 23 | 29% |
| `fact_cap_brazil` | factual | 80 | 4 | 5% |
| `fact_math` | factual | 15 | 1 | 7% |
| `fact_year` | factual | 80 | 0 | 0% |
| `comp_fox` | completion | 80 | 2 | 2% |
| `comp_road` | completion | 80 | 0 | 0% |
| `comp_storm` | completion | 80 | 0 | 0% |
| `reason_syllogism` | reasoning | 80 | 11 | 14% |
| `reason_inverse` | reasoning | 80 | 5 | 6% |
| `reason_arithmetic` | reasoning | 63 | 0 | 0% |
| `code_factorial` | code | 31 | 20 | 65% |
| `code_fizzbuzz` | code | 61 | 0 | 0% |
| `summary_photosynthesis` | summary | 80 | 1 | 1% |
| `summary_gravity` | summary | 80 | 0 | 0% |

**Overall greedy agreement:** 69/1050 = **6.6%**

By prompt type:

- **code**: 20/92 = 21.7%
- **completion**: 2/240 = 0.8%
- **factual**: 30/335 = 9.0%
- **reasoning**: 16/223 = 7.2%
- **summary**: 1/160 = 0.6%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 26 | 32% |
| `fact_cap_japan` | factual | 80 | 34 | 42% |
| `fact_cap_brazil` | factual | 80 | 15 | 19% |
| `fact_math` | factual | 15 | 2 | 13% |
| `fact_year` | factual | 80 | 10 | 12% |
| `comp_fox` | completion | 80 | 5 | 6% |
| `comp_road` | completion | 80 | 3 | 4% |
| `comp_storm` | completion | 80 | 5 | 6% |
| `reason_syllogism` | reasoning | 80 | 14 | 18% |
| `reason_inverse` | reasoning | 80 | 8 | 10% |
| `reason_arithmetic` | reasoning | 63 | 5 | 8% |
| `code_factorial` | code | 31 | 23 | 74% |
| `code_fizzbuzz` | code | 61 | 6 | 10% |
| `summary_photosynthesis` | summary | 80 | 7 | 9% |
| `summary_gravity` | summary | 80 | 7 | 9% |

**Overall top-5 overlap:** 170/1050 = **16.2%**

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

**Factored:** ' ____\nA. Tokyo\nB. Kyoto\nC. Osaka\nD. Nagoya\nAnswer:\nA\n\n Which of the following statements about the geographical characteristics of Japan is incorrect?\nA. Japan is located in the northwest of Asia.\nB. Japan is located in the temperate zone.\nC. Japan is located in the eastern hemisphere.\nD. Japan is located in the low latitude zone.\n'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** " ____.\nA. Rio de Janeiro\nB. São Paulo\nC. Brasília\nD. Belo Horizonte\nAnswer:\nB\n\nThe main reason for the formation of the 'Great Wall of China' is ____.\nA. The result of the collision and compression of the Indian Ocean and the Eurasian Plate\nB. The result of the collision and compression of the African Plate and"

**Factored:** " ____.\nA. Brasilia\nB. Sao Paulo\nC. Rio de Janeiro\nD. Sao Paulo\nAnswer:\nA\n\nThe main reason for the formation of the 'Great Depression' in the United States was ____.\nA. The United States was invaded by Japan\nB. The United States was invaded by Germany\nC. The United States was invaded by the Soviet Union\nD."

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of 17 and 23, we can add the two numbers together. \n\n17 + 23 = 40\n\nSo, 17 plus 23 equals 40.'

**Factored:** ' what?\nThe sum of seventeen and twenty-three is 30.'

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** " ____\nA. 1945\nB. 1946\nC. 1947\nD. 1948\nAnswer:\nA\n\nThe main reason for the formation of the 'Silk Road' is ____\nA. The development of commodity economy\nB. The development of transportation\nC. The development of the Silk Road\nD."

**Factored:** ' the year of the 1968 Summer Olympics. The 1968 Summer Olympics were held in Berlin, Germany, from 16 to 8 August 1968. The Games were the first to be held in the city, and the first to be held in Europe. The Games were won by the Soviet Union, with 11 gold medals, 1'

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog\ndef is_anagram(word1, word2):\n    return sorted(word1) == sorted(word2)\n\nword1 = "quick brown fox jumps over the lazy dog"\nword2 = "quick brown fox jumps over the lazy dog"\n\nif is_anagram(word1, word2):\n    print("The words are anagrams.")\nelse:\n    print("The words are not anagrams'

**Factored:** ' lazy dog.  The animals are in a ______.\nA. circle\nB. square\nC. triangle\nD. rectangle\nAnswer:\nC\n\nWhich of the following statements about the relationship between the number of chromosomes in organisms and their life activities is incorrect?\nA. In organisms with a higher ratio of DNA to protein, the number of chromosomes is generally more.\nB. In organisms with a'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' I took the one less crowded one, And entered on a gleaming hall of gold, And looked down upon the crowd, And saw the King of Kings, And the Lord of Lords, And Christ, the Son of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I saw the glory of God, And I'

**Factored:** ' the one that diverged was the one that was the same. The other road diverged in a ditch, and the ditch that diverged was the one that was the same. The ditch that diverged from the one that diverged from the yellow wood was the one that was the same. The ditch that diverged from the ditch that diverged from the yellow wood was the one that was the'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** ' I was born. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of nowhere. I was born in a small town in the middle of'

**Factored:** ' the storm broke out. The storm was so strong that it was difficult to see the road. The driver of the car was driving very slowly. The car was moving very slowly. The car was moving very slowly. The car was moving very slowly. The car was moving very slowly. The car was moving very slowly. The car was moving very slowly. The car was moving very slowly. The car'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** " warm-blooded.  If the premise is true, what can be concluded?  Hint: The body's second line of defense against pathogens includes the inflammatory response. If bacteria enter the skin through a scrape, the area may become red, warm, and painful. These are signs of inflammation. Inflammation is one way the body reacts to infections or injuries. Inflammation is caused by chemicals"

**Factored:** ' warm-blooded.  If the premise is true, which one of the following is most likely to be true?\n A. A whale is a mammal. B. A whale is cold-blooded. C. A whale is cold-blooded. D. A whale is warm-blooded. D. A whale is warm-blooded.\n \nExplanation: The premise states that all mammals are warm-blood'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining.  If the statement "The ground is wet" is true, which of the following must also be true?\nA: It is raining.\nB: The ground is not wet.\nC: The ground is not wet.\nD: It is not raining. To solve this problem, we need to analyze the given statement and the options provided. The statement is: "The ground'

**Factored:** ' it is not raining. This is an example of which type of reasoning?\n A: deductive reasoning B: inductive reasoning C: reversible reasoning D: deductive reversible reasoning\nThe reasoning provided is deductive reversible reasoning. In deductive reversible reasoning, the conclusion is derived from the premises, but the premises are not necessarily true. The reasoning is valid because it follows from '

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** ' how many apples?\nAlice starts with 3 apples. She receives 5 more apples from Bob, so we add these to her initial amount:\n\n\\[ 3 + 5 = 8 \\]\n\nNext, Alice eats 2 apples. We subtract these from her current total:\n\n\\[ 8 - 2 = 6 \\]\n\nTherefore, Alice now has \\(\\boxed{6}\\) apples'

**Factored:** ' _____ apples.\nAlice initially has 3 apples. After receiving 5 more apples from Bob, she now has:\n\n3 apples + 5 apples = 8 apples\n\nAfter eating 2 apples, Alice now has:\n\n8 apples - 2 apples = 6 apples\n\nAlice now has 6 apples.'

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef factorial_iterative(n):\n    """Return n! using recursion."""\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n\ndef factorial_recursive(n):\n    """Return n! using recursion."""\n    if n'

**Factored:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\nprint(factorial(5))'

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' print(i)\n     if i % 3 == 0:\n         print("Fizz")\n     if i % 5 == 0:\n         print("Buzz")\n     if i % 15 == 0:\n         print("FizzBuzz")\n     else:\n         print(i)\n\n# Print FizzBuzz for numbers 1 to 15\nfor i in range(1, 1'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. The process is a two-step process. First, the light energy is absorbed by chlorophyll in the thylakoid membranes of the chloroplasts. The absorbed light energy is used to convert carbon dioxide and water into glucose and oxygen. The light energy is then used to convert the glucose into other'

**Factored:** ' plants convert light energy into chemical energy in the form of glucose. The process is carried out by chlorophyll molecules in the thylakoid membranes of the chloroplasts. The process is a chemical reaction that occurs in the presence of light and water. The reaction is a process of splitting water molecules into hydrogen and oxygen molecules. The hydrogen molecules are used to produce glucose m'

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' governs the motion of all objects in the universe. It is the force that causes objects to fall to the ground when dropped, and it is the force that keeps the planets in orbit around the sun. Gravity is also responsible for the motion of planets and moons in their orbits, as well as for the motion of stars and galaxies. Without gravity, the universe would be a chaotic and unstable place.\n'

**Factored:** ' holds everything together in the universe. It is the force that causes objects to move, and it is the force that causes the universe to expand and collapse. Gravity is also the force that causes objects to fall, and it is the force that causes objects to move. It is the force that causes objects to fall, and it is the force that causes objects to move. It is the force that causes'
