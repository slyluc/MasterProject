# Diversity and ecotones in a model ecosystems of sessile species

**Authors:** Markus Hofer; Kim Sneppen

**Markus Hofer:** Medical University of Vienna, Center for Medical Data Science, Section for the Science of Complex Systems, Spitalgasse 23, 1090 Vienna, Austria; and Complexity Science Hub Vienna, Josefstädter Strasse 39, 1080 Vienna, Austria

**Kim Sneppen:** Copenhagen University, Niels Bohr Institute, Blegdamsvej 17, 2100 Copenhagen, Denmark

**Correspondence:** ksneppen@gmail.com

**Journal:** Physical Review E **109**, 024309 (2024)

**DOI:** [10.1103/PhysRevE.109.024309](https://doi.org/10.1103/PhysRevE.109.024309)

Received 28 July 2023; revised 16 December 2023; accepted 30 January 2024; published 20 February 2024.

## Abstract

Sessile species compete for space and accessible light, with directed interactions evident in one species overgrowing another and with multispecies systems characterized by nontransitive relationships. Such patterns are observed in coral reefs or lichens on rock surfaces. Open systems with episodic invasions of such species have been predicted to exhibit a stable high-diversity state when the interaction probability is below a certain critical threshold. Here, we explore this metastable high-diversity state and find that the diversity in the high-diversity state scales with the square root of the system area. When introducing two different environments, we predict a hugely increased diversity along mutual environment border. Further, the presence of spatially segregated environments is predicted to allow for increased robustness of the high-diversity state.

> Converted from the supplied PDF. Page headers and footers are omitted, line wrapping is removed, and mathematical subscripts and superscripts are preserved in LaTeX. Figures and their original captions are collected below the article text. Keep the accompanying `assets` folder beside this file to display the figures.

## I. Introduction

The biological diversity observed in ecological systems often surpasses expectations from random interaction models [1]. Although incorporating predator-prey preferences into random interaction networks enhances robustness, the maximal feasible diversity is still significantly constrained [2,3]. Traditional dynamical models limit the diversity through competitive exclusion [4,5] with possible modifications due to predatory relations [3,6]. In any case, well-connected ecosystems are mathematically fragile, suggesting separation by distance as a main determinant of stability. This effect is particularly pronounced in sessile species, whose migration is limited by generational timescales.

The population and spreading dynamics of sessile species have primarily been considered for microbes [7–12] and for the mutual rock-paper-scissor-like competition between these [13,14]. In recent years a two-dimensional cellular automata model of competing sessile species was introduced [15,16]. This model was based on randomly interacting species, inspired by the behavior of lichen on a rock surface or coral communities [17,18]. While Ref. [18] explored the loss of diversity in a model system with frozen rules of interactions for a fixed set of species, Refs. [15,16] allowed for the introduction of new species. This opened for a dynamic steady state exhibiting high species diversity. Reference [19] obtained similar results with slowly evolving species instead of introducing them from outside. Noticeably, then Ref. [16] demonstrated that an occasional random break down of cyclic relationships involving four or more species created an increase in number of distinct patches formed by species in the system, which subsequently allowed for an increase in species diversity. In this context, a cyclic relationship of four species means that individuals from species i overgrow j, j overgrows k, k overgrows l, and l finally closes the cycle by being able to overgrow i.

The central parameter in the model is γ, representing the probability that species i will take over the space of species j, if these two species are brought into contact. This potential overgrowth is assigned when species are introduced and does not change with time. The model allows us to introduce new species defined by new randomly assigned interactions. The introduction rate of new species is considered infinitesimally small, as detailed in Ref. [16]. In this limit, new species are only introduced when the dynamics in the whole lattice are frozen, meaning that no species can overgrow its neighbor. This is similar to adaptive-dynamics approaches to evolution [20–22]. A key result of this model was the emergence of a stable high-diversity state when γ < $\gamma_{c}$ with a critical $\gamma_{c}$ ∼ 0.06 (for system size L = 200). Above this threshold, diversity eventually collapses to an absorbing state with only one species [16]. Noticeably, spatial separation was essential for the obtained diversity, in the sense that diversity collapsed when species were allowed to interact without spatial constraints.

Drawing inspiration from the role of ecotones in generating [23] or maintaining ([24], Table 1) species diversity, we propose a modified version of this model. Our adaptation includes two distinct ecosystems separated by a boundary, allowing us to explore the effects of ecosystem interactions, where the interactions among individual species are location dependent.

## II. Model

We model our ecosystem as a two-dimensional, square lattice of size of L × L, where individuals of different species compete for space. Each species is characterized by a node in a directed network Γ, referred to as the interaction network. In the adjacency matrix of this network, each nondiagonal edge is assigned a value of one with probability γ and zero otherwise. Thereby, the interaction network becomes a directed Erdős-Rényi network, with γ being the probability that an edge exists between two nodes. Edges in Γ indicate the possibility that an individual from the upstream species may overgrow a site occupied by a member of the downstream species. Importantly, this matrix is highly asymmetric, as it is rare that a pair of species can invade each other.

System updates occur by selecting a lattice site along with one of its four von Neumann adjacent neighbors, ensuring that the selection respects the closed boundaries of the system. One site is designated as the attacker, the other as the defender. If there is an edge in Γ from the species in the attacking site to the species in the defending site, the attacker takes over the defending site. One may here think of the attacker as a sessile species that overgrows the defender, provided that the Γ matrix allows it.

In scenarios with two interacting ecosystems, each is defined by its unique interaction network. While these networks share the same nodes, their edge structures are distinct and assumed to be uncorrelated. Each lattice site is assigned to an ecosystem and therefore an interaction network. To obtain clear results about the interaction between ecosystems, we focused on two ecosystems sharing a straight border representing the demarcation line between two different ecosystems. In any interaction event, the outcome is determined by the interaction network associated with the defending lattice site, as this is the location where the interaction occurs.

To utilize parallel updates on graphics cards, and the speedup associated with them, we update the system following Ref. [25]: Specifically, the ecosystem is segmented into 2 × 2 blocks, arranged in a chessboard pattern. In each update cycle, we randomly select a defender from each block of one color in this pattern. Subsequently, an adjacent cell to this defender is chosen as the attacker. The interaction between the attacker and defender then proceeds according to the rules defined in the interaction network.

This update method enables simultaneous updates of $L^{2}/8$ cells, ensuring that no cell is overwritten during its interaction. After each update, there is a 50% chance that the pattern is inverted. This inversion guarantees that, on average, all cells in the lattice receive an equal number of updates. By repeating this process 8 times, each cell in the lattice is, on average, updated once, which we define as one timestep.

The parallel update method is repeated until no further site can be invaded by its neighbors, meaning the spatial configuration is frozen. When such a situation arises, α new species are introduced in the system. Each new species k is introduced at a random point in space and assigned interactions with other species according to two rules. First, we assume that the new species k is able to overgrow the species existing at point of invasion e, meaning that Γ(k, e) = 1. Secondly, all interactions with all other species are assigned interactions by adding a new column and a new row to Γ with interaction Γ(k, j) = 1 with probability γ and Γ( j, k) = 1 with probability γ for each j = 1,..., D. If the interaction is not set to 1, it is = 0 by default.

During the initial simulation phase, α is set to two, driving the system into a high-diversity state. Once the high-diversity state is reached, α is reduced back to one and we record if the system collapses back to a low-diversity state. The α = 2 when species richness D = 1 is needed for avoiding an absorbing state at D = 1, while the choice of α = 1 at the high diversity makes it simpler to record instability in terms of diversity collapse.

Despite the parallel updates it can take long until the system reaches a stationary state. This is in large part due to cyclical interactions between species. We follow Ref. [16] in speeding up these cyclical interactions until termination:

If no stationary state has been reached after $\tau_{\min}$ timesteps since the introduction of a new species, we determine all species capable of invading their neighbors. Out of these species, we select one randomly and remove all outgoing links in Γ. This ensures that the selected species is no longer competitively dominant to any other species, and as such the cyclical interaction is turned in a hierarchical one. In Ref. [16] $\tau_{\min}$ above 400 timesteps caused no noticeable change in system behavior. Despite the parallel updating method, our system shows a similar lower bound. Consequently, we conducted all simulations with a $\tau_{\min}$ of $10^{4}$ timesteps. If the removed links cause the system to reach a frozen state then the removed links in Γ are reintroduced. If the system has not reached such a frozen state after another $\tau_{\min}$ timesteps, the outgoing links of another active species are removed. This process is repeated until a stationary state is reached.

## III. Results

First, we observe the patchwork structure of different populations/species in the high-diversity state, displayed in Fig. 1(a). Here, diversity D is defined simply as the number of different species. In this picture, the majority of boundaries are frozen, as the adjacent species i and j have no edge between them in Γ. As the two species cannot invade each other, a long-lived boundary between them is formed. However, some neighboring species may overgrow each other, leading to the hierarchical elimination of some species, or to cyclic competition among three or more species. In the model section, we describe how to speed up this cyclic competition until only frozen boundaries remain.

To compare our implementation to the previous research, i.e., Ref. [16], we plot the time evolution of a system with L = 400 in Figs. 1(b) and 1(c). There, each $t_{\mathrm{ss}}$ corresponds to a state with only frozen boundaries. The two panels show, respectively, the number of different species (diversity D), and the number of distinct patches (P). A patch is defined as a connected region of the lattice that is occupied by one species, and surrounded by different species. One observes that the number of patches P easily exceeds D by a factor 10, reflecting a distribution of each species in several separate patches. These patches are typically caused by the collapse of transient cycles among four species [16] (i → j → k → m → i), that collapse to two species (i, k) that cannot invade each other. Such collapse happens when one species by random is eliminated from a part of the lattice. This then, in turn, allows the directly downstream species to grow and eliminate its “prey,” leading to a frozen configuration with two species that cannot invade each other [16].

The time series in Figs. 1(b) and 1(c) show the transient dynamics for systems with different values of γ. Each system is initialized completely empty with α set to 2 for the first 5000 stationary states. Afterward, α is set back to 1. For γ < 0.07 the high-diversity state remains stable, while higher values of γ lead to a collapse into a state with one species covering the entire lattice. The critical value of γ = $\gamma_{c}$ ≈ 0.07 and the metastability of the high-diversity state around $\gamma_{c}$ is consistent with earlier findings of Ref. [16], here with slightly higher $\gamma_{c}$ for the larger L = 400 system with closed boundaries.

To further analyze the diversity and number of patches, we plot their time averages as a function of γ in Figs. 2(a) and 2(b). In systems driven to a high-diversity state (α = 2, green dots) the average diversity and the number of patches decreases smoothly as γ is increased. Here we exclude the initial phase, where diversity and number of patches are increasing [see Figs. 1(b) and 1(c)] from the average. For the reduced introduction rate α = 1 (black crosses) the ⟨D⟩ and ⟨P⟩ follow the green dots for subcritical values of γ. However, for larger γ > $\gamma_{c}$ the diversity collapses. Thereby the plot reflects the hysteresis above $\gamma_{c}$ and also that the high-diversity state loses robustness against collapse.

Figure 3(a) demonstrates that ⟨D⟩ increases linearly with linear dimension L of the lattice, with a prefactor that decreases with γ. Thus diversity increases with the square root of the area ($L^{2}$) of the modeled ecosystem, which qualitatively is consistent with the sublinear species-area relationships, that has been reported in literature [26,27]. In our model, the diversity is supported by a number of separate patches that are independent of γ, as displayed in Fig. 3(b). Thereby, larger γ are associated with fewer coexisting species, where each is distributed into a larger number of distinct patches. Noticeably, both P and D approach 0 when L decreases below ∼ 150 [16]. The reason for this constraint is that one needs a finite diversity of about ∼ 20 to obtain sustainable diversity; for lower diversity, a single species is often able to expand across the entire lattice and the system collapses into the low-diversity state.

In Fig. 3(c), we examine the probability that two lattice sites, separated by a Manhattan distance d, are occupied by the same species. For lower values of γ, this probability exhibits an exponential decay, indicating a reduced likelihood of distant sites being occupied by the same species. However, as γ increases, this dropoff in probability occurs more gradually. When sampling unstable values of γ from the (α = 2) phase, a different behavior is observed. Moreover, the highest γ value we categorized as stable displays similar characteristics, potentially implying that a system with γ = 0.07 might experience collapse during an extended simulation period.

Systems with two sets of interaction rules are examined in Fig. 4. There the two resulting subecosystems interact at a boundary, that splits the system into a left and a right part. Panel (a) presents a system snapshot, displaying a pronounced discontinuity at the boundary, akin to an ecotone as described in ecological studies [28]. Inspired by Ref. [28] we investigate the diversity in this boundary area, measuring diversity as the number of species along subsequent columns parallel to the border, and similarly for the number of patches. As depicted in Fig. 4(b), the diversity near the boundary is significantly higher than in areas further away, consistent across all interaction probability γ levels. Both species diversity (D) and patch number (P) reach their maximum at the boundary and decline sharply with increasing distance. However, as one moves further from the boundary, this decline becomes more gradual, particularly evident in the patch number at high γ values. One also sees that the presence of the left subsystem leads to a noticeable increase in both diversity and number of patches in a substantial part of the right subsystem (and reversed). Thus the mutual synergistic effects of the systems are felt far beyond the boundary.

The interaction between the two ecological ecosystems even has an impact on the overall diversity and its robustness. This is illustrated in Fig. 5, where we see that the critical $\gamma_{c}$ changes from 0.07 to 0.10 when the ecosystem is split into two parts. One observes that the lowest high-diversity state in the uniform state has ⟨D⟩ ∼ 90 while the lowest high-diversity state in the split system has ⟨D⟩ ∼ 50 (value of blue cross at γ = 0.10). One thus concludes that the split ecosystem can sustain a high-diversity state with much fewer species than the uniform ecosystem. In fact, at high γ the effect of the split ecosystem has a profound global effect across the total system, which becomes able to sustain much larger diversity than one would have guessed by just adding two independent systems that each only allow for very low species diversity.

## IV. Discussion

The current paper analyzes a simple model for sessile species with the property that it sustains many competing species, even when random matrix considerations [1,2] would suggest this to be impossible. The model considers large systems with many species that interact randomly with each other, in a way that would lead to the extinction of all but one species if we disregarded their distribution in space [15]. Thus space is an essential part of the obtained diversity because the protection of one species from another species by barriers formed by a third bystander species is an inherent property of the model. This protection diminishes when γ increases, or if one allows individuals to give offspring far from their location, and as a consequence, the high-diversity state eventually collapses. The diversity in our model is not caused by externally imposed heterogeneity [29], but rather by species that generate a spatial heterogeneity that allows for more species. Noticeably, the obtained diversity needs space to be sustainable but needs transient cyclic interactions to be generated [16]. If we repeated any of the above calculations but only allowed the introduction of species that did not form cyclic interactions, then diversity would stay close to one. The overall dynamics is that occasional species form transient cycles, which subsequently break into man patches (heterogeneity), which in turn each allows for the introduction of a new species [16].

The relative competitive relationship between the species is assumed to be random, and quite dilute in the sense that γ is on the scale of ∼ 5%. For corresponding system diversity of about 200 this still means that each species could overgrow ∼ 10 other species present somewhere in the system at any given time. However, due to spatial constraints, many species are implicitly protected from each other by passive bystander species. The overall patchwork of species thereby becomes sustainable, and even robust to introductions of new species. Whether this is biologically reasonable depends on the types of ecosystem under consideration, with lichen communities on rocks probably being closest to fulfilling our basic assumptions of local dispersion.

Our model predicts a diversity that scales with the square root of the area:

$$
D \propto \mathrm{Area}^{0.5}, \tag{1}
$$

a sublinear relationship that is reported with various power-law exponents in the substantial literature on species-area relationships, starting with the classical paper of Arrhenius [26] and with a recent scaling exponent of about 0.2 for plant species versus the area reported in Ref. [30]. In our case, species are distributed in patches, which in turn allows one patch to be overgrown by a new species without necessarily reducing diversity. Noticeably, the number of patches is independent of the density of interactions, which in turn implies that systems with relatively high γ ∼ $\gamma_{c}$ will have relatively “few” species, D ∼ 100, distributed among more but smaller patches. This also means that the longer-range correlations for large γ ∼ $\gamma_{c}$ of Fig. 3(c) are obtained for species that each is distributed across multiple separate patches.

Our model ecosystem consists of individuals that only spread to neighbor sites, thereby simplifying the possibility of long-range dispersal used in classical models of trees [31]. In contrast to neutral models [32,33] where the fate of individual population would be governed by variants of critical branching processes, we here deal with species with fitnesses that are context dependent. In particular, the success of a given subpopulation of a species (a patch) is dependent on the properties of neighbors in space. In this perspective, one may see our interaction matrix as an idealized spatial implementation of Ref. [34]. One prediction of our model is a stronger dependence of species richness with the area than observed for real forest [30]. This may be caused by a combination of our assumption of nearest-neighbor spreading of all species, contrasting real seed dispersal that varies hugely between different types of vegetation [35].

The review by Ref. [28] discusses “ecotones,” defined as stress lines in ecological systems, that separate two different ecological regions. This “transitional area between two ecological communities” is suggested to provide special opportunities. As a consequence, ecotones often harbor more species than core areas of ecosystems, and may further facilitate the development of new traits or species [36]. Ecotones are often observed by a substantial change in vegetation patterns, caused by environmental factors like humidity or soil quality. Models of vegetation changes across ecotones based on different availability of water and nitrogen have been proposed [37], but they include so many parameters and assumptions that general lessons are difficult to extract.

Our paper offers an idealized in silico perspective on ecotones in systems ruled by sessile species. In our system, the ecotone causes increased diversity because a species may obtain refuge from an attacker on one side of the boundary by having part of its population on the other side of the boundary where they are protected by different interaction rules. Thus it is not the one-dimensional boundary that helps, but rather the bet-hedging aspect associated with subdividing its populations into regions with different rules of engagement. Our simulations demonstrated that two coupled systems may allow for many more species at substantially larger interaction frequencies than a single system. Importantly, if one simply separated the 400 × 400 system into two halves that did not interact, there would not be any substantial diversity for γ > 0.07. Our simulation shows that invasion across a common boundary supports high-diversity states up to an interaction probability γ ∼ 0.10. Thus the coupling of systems makes the whole substantially richer than the sum of its parts.

## Acknowledgment

Markus Hofer acknowledges support by the Austrian Science Fund (FWF) Grant No. P34994.

## Figures

### Figure 1

![Figure 1](assets/PhysRevE_109_024309/figure_1.png)

**Figure 1.** (a) Example state with L = 400 and γ = 0.05 (b), (c) Diversity D and number of patches P for a system with L = 400. For the first 5000 stationary states, two species are introduced. Afterwards one new species is introduced.

### Figure 2

![Figure 2](assets/PhysRevE_109_024309/figure_2.png)

**Figure 2.** (a), (b) Average diversity ⟨D⟩ and number of patches ⟨P⟩ for a system with L = 400 and γ ∈ [0.02, 0.1]. Green dots mark values measured for α = 2. Black crosses indicated the values observed for α = 1. (c) Example states, showing a complete transformation of the structure within 6 stationary states for a supercritical γ of 0.1. Only every other stationary state is displayed.

### Figure 3

![Figure 3](assets/PhysRevE_109_024309/figure_3.png)

**Figure 3.** Average diversity ⟨D⟩ (a) and average number of patches ⟨D⟩ (b) as function of system size L and interaction-probability γ. (c) Probability that two lattice sites separated by Manhattan distance d [where d = |$x_{1}$ − $x_{2}$| + |$y_{1}$ − $y_{2}$| for lattice points ($x_{1}$, $y_{1}$) and ($x_{2}$, $y_{2}$)] are occupied by the same species, sampled from 3000 static states in a system of L = 400.

### Figure 4

![Figure 4](assets/PhysRevE_109_024309/figure_4.png)

**Figure 4.** (a) Example state of a model ecosystem with L = 400 and two interaction networks. The left 200 columns in the lattice are governed by interaction network $\Gamma_{A}$ and the right 200 columns by $\Gamma_{B}$. Both interaction networks are generated with an interaction probability of γ = 0.05. (b),(c) Columnwise diversity and the average number of patches of an ecosystem of size 400 × 400 with two interaction networks. Only stable values of γ are displayed.

### Figure 5

![Figure 5](assets/PhysRevE_109_024309/figure_5.png)

**Figure 5.** Comparison between average diversities of the unmodified ecosystem and an ecosystem composed of two subecosystems as displayed in Fig. 3. Both the unmodified ecosystem and the split ecosystem have system size of 400 × 400. In total, the results from 40 simulations are displayed, the majority of data points however overlap each other. The vertical gray and blue lines indicate the presumed critical values of the unmodified and the split system respectively.

## References

[1] R. M. May, Will a large complex system be stable? Nature (London) 238, 413 (1972).

[2] S. Allesina and S. Tang, Stability criteria for complex ecosystems, Nature (London) 483, 205 (2012).

[3] J. O. Haerter, N. Mitarai, and K. Sneppen, Food web assembly rules for generalized Lotka-Volterra equations, PLoS Comput. Biol. 12, e1004727 (2016).

[4] G. F. Gause, Experimental analysis of Vito Volterra’s mathematical theory of the struggle for existence, Science 79, 16 (1934).

[5] G. Hardin, The competitive exclusion principle: An idea that took a century to be born has implications in ecology, economics, and genetics, Science 131, 1292 (1960).

[6] S. A. Levin, Community equilibria and stability, and an extension of the competitive exclusion principle, Am. Naturalist 104, 413 (1970).

[7] E. Ben-Jacob, O. Schochet, A. Tenenbaum, I. Cohen, A. Czirok, and T. Vicsek, Generic modelling of cooperative growth patterns in bacterial colonies, Nature (London) 368, 46 (1994).

[8] T. Sams, K. Sneppen, M. H. Jensen, C. Ellegaard, B. E. Christensen, and U. Thrane, Morphological instabilities in a growing yeast colony: Experiment and theory, Phys. Rev. Lett. 79, 313 (1997).

[9] O. Hallatschek and D. R. Nelson, Gene surfing in expanding populations, Theor. Popul. Biol. 73, 158 (2008).

[10] S. T. Abedon and J. Yin, Bacteriophage plaques: Theory and analysis, Methods Mol Biol. 501, 161 (2009).

[11] S. Heilmann, K. Sneppen, and S. Krishna, Coexistence of phage and bacteria on the boundary of self-organized refuges, Proc. Natl. Acad. Sci. USA 109, 12828 (2012).

[12] N. Mitarai, S. Brown, and K. Sneppen, Population dynamics of phage and bacteria in spatially structured habitats using phage λ and escherichia coli, J. Bacteriol. 198, 1783 (2016).

[13] B. Kerr, M. A. Riley, M. W. Feldman, and B. J. Bohannan, Local dispersal promotes biodiversity in a real-life game of rock–paper–scissors, Nature (London) 418, 171 (2002).

[14] T. Reichenbach, M. Mobilia, and E. Frey, Mobility promotes and jeopardizes biodiversity in rock–paper–scissors games, Nature (London) 448, 1046 (2007).

[15] J. Mathiesen, N. Mitarai, K. Sneppen, and A. Trusina, Ecosystems with mutually exclusive interactions self-organize to a state of high diversity, Phys. Rev. Lett. 107, 188101 (2011).

[16] N. Mitarai, J. Mathiesen, and K. Sneppen, Emergence of diversity in a model ecosystem, Phys. Rev. E 86, 011929 (2012).

[17] J. Jackson and L. Buss, Alleopathy and spatial competition among coral reef invertebrates, Proc. Natl. Acad. Sci. USA 72, 5160 (1975).

[18] R. H. Karlson and J. B. Jackson, Competitive networks and community structure: A simulation study, Ecology 62, 670 (1981).

[19] N. Mitarai, E. Heinsalu, and K. Sneppen, Speciation, diversification, and coexistence of sessile species that compete for space, PLoS ONE 9, e96665 (2014).

[20] G. Meszéna, M. Gyllenberg, F. J. Jacobs, and J. A. J. Metz, Link between population dynamics and dynamics of Darwinian evolution, Phys. Rev. Lett. 95, 078105 (2005).

[21] U. Dieckmann and R. Law, The dynamical theory of coevolution: a derivation from stochastic ecological processes, J. Math. Biol. 34, 579 (1996).

[22] K. Sneppen, Extremal dynamics and punctuated co-evolution, Physica A 221, 168 (1995).

[23] T. B. Smith, R. K. Wayne, D. J. Girman, and M. W. Bruford, A role for ecotones in generating rainforest biodiversity, Science 276, 1855 (1997).

[24] S. Kark and B. J. Van Rensburg, Ecotones: Marginal or central areas of transition? Isr. J. Ecol. Evol. 52, 29 (2006).

[25] F. Uekermann, J. Mathiesen, and N. Mitarai, Characterization of phase transitions in a model ecosystem of sessile species, Phys. Rev. E 95, 032409 (2017).

[26] O. Arrhenius, Species and area, J. Ecol. 9, 95 (1921).

[27] J. Dengler, Which function describes the species–area relationship best? A review and empirical evaluation, J. Biogeo. 36, 728 (2009).

[28] K. Hufkens, P. Scheunders, and R. Ceulemans, Ecotones in vegetation ecology: methodologies and definitions revisited, Ecol. Res. 24, 977 (2009).

[29] M. L. Rosenzweig, Species diversity in space and time (Cambridge University Press, 1995).

[30] W. M. Lonsdale, Global patterns of plant invasions and the concept of invasibility, Ecology 80, 1522 (1999).

[31] J. Chave and E. G. Leigh Jr, A spatially explicit neutral model of β-diversity in tropical forests, Theor. Popul. Biol. 62, 153 (2002).

[32] J. Chave, Neutral theory and community ecology, Ecol. Lett. 7, 241 (2004).

[33] J. Rosindell, S. P. Hubbell, F. He, L. J. Harmon, and R. S. Etienne, The case for ecological neutral theory, Trends Ecol. Evol. 27, 203 (2012).

[34] W. D. Kissling, C. F. Dormann, J. Groeneveld, T. Hickler, I. Kühn, G. J. McInerny, J. M. Montoya, C. Römermann, K. Schiffers, F. M. Schurr et al., Towards novel approaches to modelling biotic interactions in multispecies assemblages at large spatial extents, J. Biogeo. 39, 2163 (2012).

[35] M. C. Côrtes and M. Uriarte, Integrating frugivory and animal movement: a review of the evidence and implications for scaling seed dispersal, Biolog. Rev. 88, 255 (2013).

[36] M. Schilthuizen, Ecotone: speciation-prone, Trends Ecol. Evol. 15, 130 (2000).

[37] D. P. Peters, Plant species dominance at a grassland - shrubland ecotone: an individual-based gap dynamics model of herbaceous and woody species, Ecol. Model. 152, 5 (2002).
