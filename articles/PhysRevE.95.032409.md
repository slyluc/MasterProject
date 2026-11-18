# Characterization of phase transitions in a model ecosystem of sessile species

**Authors:** Florian Uekermann, Joachim Mathiesen, and Namiko Mitarai

**Affiliation:** Niels Bohr Institute, Copenhagen University, Blegdamsvej 17, 2100 Copenhagen, Denmark

**Correspondence:** mitarai@nbi.dk

**Journal:** Physical Review E **95**, 032409 (2017)

**DOI:** [10.1103/PhysRevE.95.032409](https://doi.org/10.1103/PhysRevE.95.032409)

Received 1 November 2016; published 13 March 2017.

## Abstract

We consider a model ecosystem of sessile species competing for space. In particular, we consider the system introduced by Mathiesen et al. [J. Mathiesen, N. Mitarai, K. Sneppen, and A. Trusina, Phys. Rev. Lett. 107, 188101 (2011)] where species compete according to a fixed interaction network with links determined by a Bernoulli process. In the limit of a small introduction rate of new species, the model exhibits a discontinuous transition from a high-diversity state to a low-diversity state as the interaction probability between species, γ, is increased from zero. Here we explore the effects of finite introduction rates and system size on the phase transition by utilizing efficient parallel computing. We find that the low state appears for γ > $\gamma_{c}$. As γ is increased further, the high state approaches the low state, suggesting the possibility that the two states merge at a high γ. We find that the fraction of time spent in the high state becomes longer with higher introduction rates, but the availability of the two states is rather insensitive to the value of the introduction rate. Furthermore, we establish a relation between the introduction rate and the system size, which preserves the probability for the system to remain in the high-diversity state.

> Converted from the supplied PDF. Page headers and footers are omitted, line wrapping is removed, and mathematical subscripts and superscripts are preserved in LaTeX. Figures and their original captions are collected below the article text. Keep the accompanying `assets` folder beside this file to display the figures.

## I. Introduction

Many ecosystems maintain a high diversity in spite of the competition among species, and mechanisms of coexistence remain a central question in ecology. In particular, the seminal works by Gardner and Ashby [1] and May [2] that pointed out in a simple toy model that a well-mixed system with random interaction inhibits stable coexistence made researchers explore various explanations for the development of stable ecosystems [3–6]. For example, a nonhierarchical interaction or cyclic relationship among species has been proposed as a way to obtain long-lasting coexistence [7–11], though it often causes oscillatory dynamics of the population. When space is explicitly considered, the oscillation of the total population can be suppressed and instead gives rise to a spiral wavelike dynamics, which further stabilizes the coexistence of species [12–27].

In [28] a simple spatial ecosystem model was introduced for mutually exclusive and sessile species that compete for available space. The model was inspired by the coexistence of lichen species [29–32] and corals [14,15,33,34]. The model introduced a competitive interaction between occupants of neighboring sites on a two-dimensional square lattice. A species can replace the occupant of a neighboring site if a predefined interaction network contains a link from that species to the species in the neighboring site. In the model, the interactions among species are assigned randomly, according to a Bernoulli process. When the connectivity of the interaction network increases, a transition from multiple coexisting species to one species is observed [28]. For this model it was shown that the presence of cyclic interactions among species is necessary to maintain a high-diversity state [35]. In particular, cyclic interactions lead to a complex spatiotemporal dynamics that creates spatial patches, which provide niches for a newly introduced species. In the limit of small introduction rates, the transition between the high- and low-diversity states was found to be sharp and discontinuous. For small, but finite, introduction rates the model shows a bistable dynamics, switching between low- and high-diversity states above the critical connectivity [35]. This behavior is found to be robust to changes in the lattice structure [28], the introduction of a very small death rate [36], and to making the introduced new species mutations of existing ones [37]. For larger introduction rates, the diversity tends to be higher and the bistability is smeared out by inherent noise in the dynamics.

This is one of the simplest models of spatial coexistence of competitive species. The interaction network is completely random, and the model is characterized by the only two parameters, i.e., the connectivity of the interaction network and the introduction rate of new species. Understanding the properties of the transition between low- and high-diversity states will improve our basic understanding of the coexistence mechanism. However, previously, the characteristics of the transition were not carefully analyzed. In particular, it was unclear whether separate high- and low-diversity states exist for large introduction rates, or if they merge at a critical point.

Here we examine the effect of the species introduction rate on the transition between low and high diversity. We first demonstrate that increasing the introduction rate gradually reduces the time spent in the low-diversity state, while the bistable dynamics is maintained. However, independent of the introduction rate, the model maintains a qualitative transition at a critical probability of interactions, below which the low-diversity state is inaccessible. The distinction between the high- and low-diversity states tends to be smaller with higher interaction probability, again independent of the introduction rate. Thus, the introduction rate and frequency of interaction play orthogonal roles in how they affect the qualitative dynamics. The accessibility of the low-diversity state is determined by the frequency of interaction, while the introduction rate controls the time spent in the respective state if the low-diversity state is accessible. Finally, we show that, scaling the introduction rate inversely proportional to the lattice length, the fraction of time spent in high and low states is conserved for different system sizes.

The paper is structured as follows: In Sec. II, we briefly describe the model [28,35] and the algorithm for parallel computing. We also give precise definitions of the terms introduction rate and interaction frequency. Section III A introduces the frozen boundary length as a more robust classification criterion for low and high states, compared to the diversity. In Secs. III B and III C, we examine how the interaction frequency, introduction rate, and system size influence the availability and prevalence of the low state. Finally, we observe a continuous behavior at the phase transition for finite introduction rates.

## II. Model

In this simple ecosystem model, multiple species compete for occupation of space on a two-dimensional square lattice. A lattice site can only be occupied by one species at a time. The interactions in the ecosystem are defined by a directed network of competitive interactions between species. A species can invade a neighboring site and replace its occupant if the interaction network contains a link from the invading species to the occupying species. The interaction network is described by Γ($s_{i}$,$s_{o}$) ∈ {0,1}, where $s_{i}$ and $s_{o}$ represent the invading species and occupying species, respectively. Γ($s_{i}$,$s_{o}$) = 1 describes a link from species $s_{i}$ to $s_{o}$, which means $s_{i}$ can replace $s_{o}$ on a neighboring lattice site. Each element of the matrix Γ is predetermined to be 1 with probability γ, or zero with probability 1 − γ. In this model, there are two ways for species $s_{i}$ and $s_{j}$ to be competitively equivalent: One way is Γ($s_{i}$,$s_{j}$) = Γ($s_{j}$,$s_{i}$) = 1, i.e., the species can replace each other; the other is Γ($s_{i}$,$s_{j}$) = Γ($s_{j}$,$s_{i}$) = 0, i.e., they cannot replace each other (standoff). Note that both cases are observed in nature; mutual active competition was reported in epifaunal communities [38], while for crustose lichens, a contact boundary is formed at the encounter of one with another, and if they are competitively neutral these boundaries may remain stable over time [32].

Previous implementations [28,35] used a random sequential update scheme, consisting of repeatedly picking two neighboring sites as source and target of a potential invasion. To utilize the highly parallel architecture of modern graphics processing units, we here use a modified updating scheme. The square lattice is divided into two sets of 2 × 2 chunks. Each set consists of all chunks that are not directly adjacent to each other, resembling the set of black and white sites of a checkerboard. In each chunk of one set a random site is picked as the target for the potential invasion and a random neighbor is picked as the source. If the interaction network contains a link from the invading species to the one occupying the target site, the invading species becomes the occupant of the target site. This update step is repeated alternatingly for the two sets of chunks (black and white). This update mechanism was chosen to allow parallel updates that guarantee that the origin of an invasion is never invaded at the same time [graphics processing unit (GPU) friendly], while preserving a local dynamics similar to random sequential updates. We have found no qualitative or quantitative differences between these simulations and those with random sequential updates.

The rate of introducing new species is parametrized by the introduction rate α. A new species is introduced at a random site with a rate of $\alpha L^{-2}$, where L represents the length of the lattice. For each potential interaction between the new and existing species a link is added with probability γ. Additionally the interaction network will always contain a link from the new species to the former occupant of the site at which it is introduced. This definition of α differs from our previous papers [28,35], where we chose α to be the rate of attempted introductions of a new species, which were rejected if the interaction network did not have a link from the new species to the old one. Thus, the α value used in this paper would have to be divided by γ when one compares it with previous results.

## III. Results

### A. Diversity and frozen boundary length

In previous work the diversity of species, defined as the number of species, was used as the main observable to quantify the state of the system. The mean diversity shows a sharp transition from high- to low-diversity values at γ ≈ 0.055 when α is close to zero and N = $192^{2}$ [28,35] as shown in Fig. 1(a). However, the fluctuation of the diversity turned out to be significant for larger α [see Fig. 3(a), green dashed line], making the classification of the two states difficult.

It was found that the transition to high- or low-diversity state is preceded by an increase or decline in the number of patches (spatially connected sites occupied by the same species) [35], which can also be used as an indicator of the high and low states. Among those patches, the most important ones for high diversity are the patches surrounded by the species with standoff relations, since those patches will be long lasting. This can be clearly seen in Fig. 2, where the active boundaries between interacting species and the frozen boundaries between standoff relation species are marked in the snapshots of the high and low states. We found that the length of boundaries between sites with different species of standoff relationships [Fig. 3(a), blue dotted line] is a better indicator of the high and low states with less fluctuation than the diversity [Fig. 3(a), green dashed line] and shows the transition between the high and the low states at the same position as the mean diversity for small α [Fig. 1(b)]. We therefore use this quantity, normalized with its maximum possible value ($2L^{2}$), to study the transition in finite α and call it the frozen boundary length $F_{l}$.

### B. High state and low state

Figure 4(a) shows the distribution of the frozen boundary length $F_{l}$ for different introduction rates α at γ = 0.12. With growing α, the distribution gains a high mode. For lower values of α, the peak at large $F_{l}$ lowers its height and eventually becomes undetectable. For high α, the peak at low $F_{l}$ becomes undetectable. For intermediate values (dashed grey line), the distribution is bimodal. This is consistent with the observation in previous work that in the limit of α → 0 no bistability is observed and the system is exclusively in a low state when γ exceeds the critical value $\gamma_{c}$.

For a fixed α, the peak position of the high mode decreases to $F_{l}$ ≈ 0 with increasing γ as shown in Fig. 4(b), while the left peak in low $F_{l}$ stays always around $F_{l}$ = 0 if it is there. Eventually, for high enough γ the two peaks becomes indistinguishable within our numerical accuracy. Figure 4(b) also shows that the high peak’s position is rather insensitive to the system size L and the value of α.

In the following we use the location of the peak of the right mode of the frozen boundary length m of a given γ and L with α = 0.256 to classify whether the system is in a high or low state. More specifically, we record the time series of the frozen boundary length $F_{l}$ [e.g., Fig. 3(a), blue dotted line], and if $F_{l}$ is longer than 0.5m, then the system is classified to have entered a high state of samples, until the length of the boundary drops below 0.1m and vice versa. This allows us to classify the state of the system, as shown by the red curve in Fig. 3(a).

### C. Fraction of time in high and low states

Using the frozen boundary length as a classification criterion as described in the previous section, we classified the samples of simulations with different choices for L ∈ {192,256,320} and various α and γ. Figure 5 shows our results for L = 320.

As observed previously [35], the clearly separated low state becomes available at γ > $\gamma_{c}$ when α is small enough. The critical value was found to be $\gamma_{c}$ ≈ 0.055, for L = 192 [28,35]. For larger systems L ∈ 256,320 we find larger critical values $\gamma_{c}$ ≈ 0.1.

With increasing γ beyond $\gamma_{c}$, the fraction of time spent in a high state approaches zero in a continuous fashion, and the fraction increases with α as shown Fig. 5(a). We observe that the fraction of time spent in a high state approaches a steplike shape as a function of γ with a sharp change at $\gamma_{c}$ as the limit α → 0 is approached, which is consistent with the results of quasistatic simulations [28] that showed a discontinuous transition.

When plotting the time spent in a low state over α for fixed γ [Fig. 5(b)], we find that with increasing α the time in the low state decreases faster than exponentially at first, but seems to approach an exponential decline as α increases. This behavior is observed for all γ > $\gamma_{c}$. At lower γ we did not find any low-state samples regardless of α.

In practice we can measure this decline of time spent in a low state only over a few orders of magnitude (two to four), due to limitations on simulation time. Consequently, the range of α values where we can measure this exponential decline is limited. This applies especially at low γ values, which exhibit quicker decline (larger absolute exponent). However, our observations do not indicate any limit to the exponential decline with increasing α values. This suggests that the γ value governs whether the low state is available at all (γ > $\gamma_{c}$), while α is an orthogonal parameter, which determines how much time the system spends in a low state if γ > $\gamma_{c}$.

For different system sizes we observe the same qualitative behavior (we tested L = 192, L = 256, and L = 320). However, we observe that α needs to be scaled inversely proportional to the system size to achieve a quantitatively similar amount of time spent in low and high states (see Fig. 6).

The scaling of α with L can be understood as follows: In a low state, typically there is only one species in a system. When a new species is introduced, it will take time proportional to L for a new species to replace the old one. If the next new species is introduced after this displacement, there is no chance for the system to go to a high state. In other words, αL determines the rate for the system state to switch from low to high.

## IV. Discussion

Through extensive numerical simulations, we have considered the transition between high- and low-diversity states in a model ecosystem. The simulations show that there is a clear discontinuous transition at γ = $\gamma_{c}$, above which the low state, clearly separated from the high state, becomes available independent of the value of α. In the α → 0 limit, the system stays in the low state for γ > $\gamma_{c}$, but with finite α, we have confirmed the bistability between the high state and the low state.

Interestingly, the high state seems to approach the low state as γ approaches 1, rather independently from α. It is possible that the high state appears continuously from the low state at a certain γ. When γ = 1, i.e., when every species can invade everyone else, the frozen boundary length must be zero and there should not be any distinction between the high state and the low state. It would be interesting to confirm at which value of γ the high state appears, when decreasing γ from 1; it may continuously appear at γ = 1, or there there may be another transition to the high state at γ < 1. In the latter case, the high state may appear continuously from the low state, discontinuously with small separation between the two states. This is difficult to confirm numerically, because of fluctuations and the small gap between the low and the high states at high γ we see in our simulations. It should also be noted that our result cannot exclude some dependency on α of the high state at high γ, since identification of the high peak is less accurate at high γ ’s. Further investigation of the nature of this transition likely depends on the development of analytical insight.

In the present work, we could not identify a critical point where the two states merge, which would have been expected in a more conventional phase transition. Further research is needed to clarify the nature of the transition in the present model.

Overall, our result suggests that the available states are solely controlled by γ, while α controls the fraction of time spent in each state. This may suggest that the available states are determined by geometrical constraint such as how many neighbor species exist and can interact for a given γ. It would be interesting to see the transition in three dimensions where the geometrical constraints are entirely different from those in two dimensions or on a different lattice. It could provide us better understanding of the mathematical structure behind the transition. It may also provide some insight about the coexistence of a microbial community, which is often studied in two-dimensional lattice models with various invasion rules [20,39] but can form three-dimensional structures in, e.g., soil and biofilms.

## Acknowledgment

This work is supported by the Danish National Research Foundation through the Center for Models of Life.

## Figures

### Figure 1

![Figure 1](assets/PhysRevE_95_032409/figure_1.png)

**Figure 1.** (a) Mean diversity and (b) mean length of frozen boundary as a function of interaction frequency γ for different introduction rates α. N = $320^{2}$.

### Figure 2

![Figure 2](assets/PhysRevE_95_032409/figure_2.png)

**Figure 2.** Snapshots of system in (a), (b) low- and (c), (d) high-diversity state. (a) and (c) show magnification of upper left corners of the snapshots (b) and (d), respectively, with marked frozen (black) and active (white) boundaries. N = $192^{2}$, γ = 0.7, and α = 0.0005.

### Figure 3

![Figure 3](assets/PhysRevE_95_032409/figure_3.png)

**Figure 3.** Time series of diversity and frozen boundary length with introduction rate α = 0.003 and interaction frequency γ = 0.16 in an N = $192^{2}$ system. For higher γ and α the diversity (green dashed line) fluctuations are stronger and the frozen boundary length (blue dotted line) is a more reliable criterion. The solid red curve shows whether the system is classified to be in the high or low state, based on the frozen boundary length. The curve shows the frozen boundary length threshold value that was crossed for the current classification.

### Figure 4

![Figure 4](assets/PhysRevE_95_032409/figure_4.png)

**Figure 4.** (a) Distribution of boundary lengths with N = $320^{2}$, γ = 0.12, for various α. (b) Location of maximum (mode) as a function of γ.

### Figure 5

![Figure 5](assets/PhysRevE_95_032409/figure_5.png)

**Figure 5.** Fraction of (a) high- and (b) low-state samples over γ and α. N = $320^{2}$.

### Figure 6

![Figure 6](assets/PhysRevE_95_032409/figure_6.png)

**Figure 6.** Phase diagram. Classification threshold for the frozen boundary is questionable in grey area due to boundary length approaching zero.

## References

[1] M. R. Gardner and W. R. Ashby, Nature (London) 228, 784 (1970).

[2] R. M. May, Nature (London) 238, 413 (1972).

[3] A. Roberts, Nature (London) 251, 607 (1974).

[4] S. L. Pimm, Nature (London) 307, 321 (1984).

[5] K. S. McCann, Nature (London) 405, 228 (2000).

[6] A. R. Ives and S. R. Carpenter, Science 317, 58 (2007).

[7] Y. Itoh, Proc. Jpn. Acad. 47, 854 (1971).

[8] R. M. May and W. J. Leonard, SIAM J Appl. Math. 29, 243 (1975).

[9] J. Hofbauer and K. Sigmund, Evolutionary Games and Population Dynamics (Cambridge University Press, Cambridge, U.K., 1998).

[10] T. Reichenbach, M. Mobilia, and E. Frey, Phys. Rev. E 74, 051907 (2006).

[11] J. C. Claussen and A. Traulsen, Phys. Rev. Lett. 100, 058104 (2008).

[12] P. Dayton, Ecol. Monogr. 41, 351 (1971).

[13] M. E. Gilpin, Am. Nat. 109, 51 (1975).

[14] J. Jackson and L. Buss, Proc. Natl. Acad. Sci. USA 72, 5160 (1975).

[15] R. Karlson and L. Buss, Ecol. Modell. 23, 243 (1984).

[16] K.-i. Tainaka, J. Phys. Soc. Jpn. 57, 2588 (1988).

[17] M. C. Boerlijst and P. Hogeweg, Phys. D (Amsterdam, Neth.) 48, 17 (1991).

[18] M. B. Cronhjort, Origins Life Evol. Biospheres 25, 227 (1995).

[19] G. Szabo and T. Czaran, Phys. Rev. E 64, 042902 (2001).

[20] B. Kerr, M. A. Riley, M. W. Feldman, and B. J. Bohannan, Nature (London) 418, 171 (2002).

[21] D. J. Murrell and R. Law, Ecol. Lett. 6, 48 (2003).

[22] B. C. Kirkup and M. A. Riley, Nature (London) 428, 412 (2004).

[23] R. Laird and B. Schamp, Am. Nat. 168, 182 (2006).

[24] B. Kerr, C. Neuhauser, B. J. M. Bohannan, and A. M. Dean, Nature (London) 442, 75 (2006).

[25] T. Reichenbach and E. Frey, Phys. Rev. Lett. 101, 058102 (2008).

[26] S. Rulands, A. Zielinski, and E. Frey, Phys. Rev. E 87, 052710 (2013).

[27] A. Szolnoki and M. Perc, Phys. Rev. E 93, 062307 (2016).

[28] J. Mathiesen, N. Mitarai, K. Sneppen, and A. Trusina, Phys. Rev. Lett. 107, 188101 (2011).

[29] P. M. Harris, Oecologia 108, 663 (1996).

[30] T. H. Nash, Lichen Biology (Cambridge University Press, Cambridge, U.K., 2008).

[31] M. Grube, M. Cardinale, J. V. de Castro, H. M ̈uller, and G. Berg, ISME J. 3, 1105 (2009).

[32] E. Jettestuen, A. Nermoen, G. Hestmark, E. Timdal, and J. Mathiesen, PLoS One 5, e12820 (2010).

[33] L. Aerts and R. Soest, Mar. Ecol. Prog. Ser. 148, 125 (1997).

[34] L. Buss and J. Jackson, Am. Nat. 113, 223 (1979).

[35] N. Mitarai, J. Mathiesen, and K. Sneppen, Phys. Rev. E 86, 011929 (2012).

[36] F. Botta and N. Mitarai, Phys. Rev. E 89, 022704 (2014).

[37] N. Mitarai, E. Heinsalu, and K. Sneppen, PloS One 9, e96665 (2014).

[38] A. M. Kay and M. J. Keough, Oecologia 48, 123 (1981).

[39] T. L. Czárán, R. F. Hoekstra, and L. Pagie, Proc. Natl. Acad. Sci. USA 99, 786 (2002).
