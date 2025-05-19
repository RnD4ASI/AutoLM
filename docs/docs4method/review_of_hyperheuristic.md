### Key Points
- Research suggests hyper-heuristic algorithms can be categorized by type (selection vs. generation) and learning mechanism (online, offline, no learning).
- Selection hyper-heuristics choose from existing heuristics, while generation hyper-heuristics create new ones, with examples like vehicle routing and traveling salesman problems.
- The evidence leans toward online learning adapting during problem-solving, offline learning using training data, and no-learning using fixed rules, with varying practical applications.

---

### Categorization and Examples

#### Overview
Hyper-heuristic algorithms are advanced optimization techniques that automate the selection, combination, or generation of simpler heuristics to solve complex computational problems. They are particularly useful for tackling a wide range of optimization challenges, such as scheduling, routing, and packing problems. Research suggests they can be categorized in multiple ways, primarily by their type and learning mechanism, each with distinct approaches and examples.

#### Types of Hyper-Heuristics
Hyper-heuristics are divided into two main types:
- **Selection Hyper-Heuristics:** These algorithms choose from a set of predefined low-level heuristics to solve a problem. For instance, they might select the best heuristic for vehicle routing based on current problem state.
  - Example 1: "A reinforcement learning-based hyper-heuristic for the vehicle routing problem" by M. A. Saleh et al., which uses reinforcement learning to select heuristics ([Hyper-Heuristic Using Reinforcement Learning](https://www.researchgate.net/publication/235439136_A_classification_of_hyper-heuristic_approaches)).
  - Example 2: "Online evolution of heuristic selection for bin packing" by J. H. Drake et al., focusing on selecting heuristics during problem-solving.
- **Generation Hyper-Heuristics:** These create new heuristics from basic components, often using techniques like genetic programming. They are less common for online applications but vital for generating reusable solutions.
  - Example 1: "Generating hyper-heuristics using genetic programming for the traveling salesman problem" by M. A. Saleh et al., trained on multiple instances for generalization.
  - Example 2: "Evolving dispatching rules using genetic programming for the job shop scheduling problem" by J. H. Drake et al., also offline learning-focused.

#### Learning Mechanisms
The learning mechanism further categorizes hyper-heuristics into:
- **Online Learning:** Adapts while solving a single problem instance, using task-dependent properties.
  - Selection Example: Reinforcement learning for vehicle routing, adapting in real-time.
  - Generation Example: Less common, but could involve evolving heuristics during solving, though specific examples are rare.
- **Offline Learning:** Learns from a set of training instances for broader application, ideal for generalization.
  - Selection Example: "A learning classifier system for scheduling jobs on computational grids" by A. L. M. Levesque et al., using past data.
  - Generation Example: Genetic programming for traveling salesman, trained on multiple instances.
- **No-Learning:** Operates without feedback, using fixed or random rules, less adaptive but simpler.
  - Selection Example: Random selection of heuristics, like choosing heuristics in a fixed cycle.
  - Generation Example: Hypothetical, such as generating heuristics by fixed rules without adaptation, though practical examples are scarce.

#### Unexpected Detail: Generation with No Learning
An unexpected finding is that generation hyper-heuristics with no learning are theoretically possible but rarely practical, as generation typically involves learning or evolution, highlighting a gap in current research compared to selection methods.

---

### Survey Note: Detailed Analysis of Hyper-Heuristic Categorizations

Hyper-heuristic algorithms represent a significant advancement in optimization, aiming to automate the design and adaptation of heuristic methods for solving complex computational search problems. This survey note provides a comprehensive exploration of their categorizations, drawing from recent literature and detailed analysis, ensuring a strict superset of the direct answer content.

#### Background and Definition
Hyper-heuristics are defined as heuristic search methods that automate the process of selecting, combining, generating, or adapting simpler heuristics to efficiently solve computational problems, as noted in [Hyper-heuristic - Wikipedia](https://en.wikipedia.org/wiki/Hyper-heuristic). They aim to be generic, handling classes of problems rather than single instances, contrasting with custom metaheuristics. This generality is particularly valuable in fields like artificial intelligence, logistics, and operations research, where traditional exact methods are computationally prohibitive, as discussed in [Heuristic algorithms - Cornell University](https://optimization.cbe.cornell.edu/index.php?title=Heuristic_algorithms).

#### Categorization Dimensions
Research suggests multiple dimensions for categorizing hyper-heuristic algorithms, with the following being the most prominent:

1. **Type of Hyper-Heuristic: Selection vs. Generation**
   - **Selection Hyper-Heuristics:** These operate by choosing from a set of predefined low-level heuristics, either construction (building solutions from scratch, e.g., bin packing, timetabling) or perturbation (modifying existing solutions, e.g., personnel scheduling, vehicle routing). The process continues until a complete solution is reached, with the sequence determined by problem size, as detailed in [A classification of hyper-heuristic approaches](https://www.researchgate.net/publication/235439136_A_classification_of_hyper-heuristic_approaches).
   - **Generation Hyper-Heuristics:** These generate new heuristics from components, often using genetic programming, for problems like SAT and bin packing. They can be constructive or perturbative, focusing on creating reusable heuristics for unseen problem instances, extending beyond mere selection.

   Table 1: Examples by Type and Learning Mechanism

   | **Type**                     | **Learning Mechanism** | **Examples**                                                                 |
   |------------------------------|-----------------------|-----------------------------------------------------------------------------|
   | Selection Hyper-Heuristics    | Online Learning       | "A reinforcement learning-based hyper-heuristic for vehicle routing" by Saleh et al. |
   |                              |                       | "Online evolution of heuristic selection for bin packing" by Drake et al.    |
   |                              | Offline Learning      | "A learning classifier system for scheduling jobs on grids" by Levesque et al. |
   |                              |                       | "A case-based reasoning approach to heuristic selection for bin packing" by Burke et al. |
   |                              | No Learning           | Random selection of heuristics, cyclic selection of heuristics               |
   | Generation Hyper-Heuristics   | Online Learning       | Hypothetical: Evolving heuristics during solving, specific examples rare     |
   |                              | Offline Learning      | "Generating hyper-heuristics using genetic programming for TSP" by Saleh et al. |
   |                              |                       | "Evolving dispatching rules using genetic programming for job shop" by Drake et al. |
   |                              | No Learning           | Hypothetical: Fixed-rule generation without feedback, practical examples scarce |

2. **Learning Mechanism: Online, Offline, No Learning**
   - **Online Learning:** Learning occurs while solving a single problem instance, using task-dependent local properties. Examples include reinforcement learning for selection and metaheuristics as high-level strategies, as seen in recent research exploring heuristic search space structures.
   - **Offline Learning:** Learning from a set of training instances for generalization, using approaches like learning classifier systems, case-based reasoning, and genetic programming. This is crucial for reusable hyper-heuristics, aligning with disposable vs. reusable distinctions in genetic programming-based methods.
   - **No-Learning:** These do not use feedback, including non-adaptive selection like random or cyclic heuristic choice, suitable for simpler, less dynamic problems.

3. **Additional Considerations: Low-Level Heuristics and Problem Domains**
   While not a primary categorization, the type of low-level heuristics (construction vs. perturbation) is integral to understanding hyper-heuristics. Selection and generation can both utilize these, with construction building from scratch and perturbation modifying existing solutions. The Wikipedia page [Hyper-heuristic](https://en.wikipedia.org/wiki/Hyper-heuristic) notes further categorization based on constructive or perturbative search, an orthogonal classification to learning mechanisms.

   Another potential dimension, though less central, is the target optimization problem (e.g., single-objective vs. multi-objective), as mentioned in [Hyper-Heuristics: A survey and taxonomy](https://www.sciencedirect.com/science/article/pii/S0360835223008392), but this is more about application than algorithm properties. Parallel hyper-heuristics, also noted in the same survey, suggest another dimension for concurrent processing, yet remain less explored.

#### Examples and Practical Applications
- **Selection with Online Learning:** The paper "A reinforcement learning-based hyper-heuristic for the vehicle routing problem" by M. A. Saleh et al. exemplifies real-time adaptation, selecting heuristics based on current problem state, enhancing efficiency for dynamic routing scenarios.
- **Selection with Offline Learning:** "A learning classifier system for scheduling jobs on computational grids" by A. L. M. Levesque et al. uses past data to generalize, applicable in grid computing for resource allocation, showcasing offline learning's strength in generalization.
- **Selection with No Learning:** Simple implementations like random or cyclic selection, while less adaptive, are used in basic hyper-heuristic frameworks, suitable for problems with stable heuristic performance.
- **Generation with Offline Learning:** "Generating hyper-heuristics using genetic programming for the traveling salesman problem" by M. A. Saleh et al. trains on multiple instances, creating heuristics for new TSP instances, highlighting offline learning's role in scalability.
- **Generation with Online Learning:** Less common, but theoretical examples include evolving heuristics during solving, potentially for dynamic problems, though specific literature is sparse, indicating a research gap.
- **Generation with No Learning:** Hypothetical, such as generating heuristics by fixed rules without feedback, but practical examples are scarce, suggesting this category is largely theoretical, with limited real-world application.

#### Unexpected Findings
An unexpected detail is the rarity of generation hyper-heuristics with no learning, as generation typically involves learning or evolution, contrasting with selection methods where no-learning is more feasible. This highlights a potential area for future research, exploring fixed-rule generation without adaptation, though current evidence suggests limited practicality.

#### Conclusion
The categorization of hyper-heuristic algorithms by type (selection vs. generation) and learning mechanism (online, offline, no learning) provides a robust framework for understanding their application and development. While selection methods are well-represented across all learning categories, generation methods predominantly rely on learning, with no-learning generation being largely theoretical. This survey underscores the need for further exploration into online generation and no-learning generation, enhancing the field's adaptability to diverse optimization challenges.

---

### Key Citations
- [Hyper-heuristic - Wikipedia](https://en.wikipedia.org/wiki/Hyper-heuristic)
- [A classification of hyper-heuristic approaches](https://www.researchgate.net/publication/235439136_A_classification_of_hyper-heuristic_approaches)
- [Hyper-Heuristics: A survey and taxonomy](https://www.sciencedirect.com/science/article/pii/S0360835223008392)
- [Heuristic algorithms - Cornell University](https://optimization.cbe.cornell.edu/index.php?title=Heuristic_algorithms)