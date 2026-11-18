# Ecosystems with mutually exclusive interactions self-organize to a state of high diversity

**Authors:** Joachim Mathiesen, Namiko Mitarai, Kim Sneppen and Ala Trusina

**Affiliation:** Niels Bohr Institute / CMOL, Blegdamsvej 17, DK-2100 Copenhagen, Denmark

**Version:** arXiv:1108.2011v1 [q-bio.PE], 8 Aug 2011

## Abstract

Ecological systems comprise an astonishing diversity of species that cooperate or compete with each other forming complex mutual dependencies. The minimum requirements to maintain a large species diversity on long time scales are in general unknown. Using lichen communities as an example, we propose a model for the evolution of mutually excluding organisms that compete for space. We suggest that chain-like or cyclic invasions involving three or more species open for creation of spatially separated sub-populations that subsequently can lead to increased diversity. In contrast to its non-spatial counterpart, our model predicts robust co-existence of a large number of species, in accordance with observations on lichen growth. It is demonstrated that large species diversity can be obtained on evolutionary timescales, provided that interactions between species have spatial constraints. In particular, a phase transition to a sustainable state of high diversity is identified.

> Converted from the supplied PDF. Page headers and footers are omitted, line wrapping is removed, and mathematical subscripts and superscripts are preserved in LaTeX. Figures and their original captions are collected below the article text. Keep the accompanying `assets` folder beside this file to display the figures.

## Introduction

Interactions between biological species may well be as old as life itself [1–3] with competition and predation as major determinants for species diversity [4, 5]. Competitive exclusion [6, 7] has been suggested to reduce ecosystem diversity when several species compete for the same resources. Real ecosystems, on the other hand, consist of multiple species and have a robustness that may even increase with diversity [8, 9]. To obtain robustness of ecosystem diversity in theoretical models one needs first of all to limit the exponential growth by assuming a maximum carrying capacity for the population of each species [10]. Extreme version of such models [11, 12] indeed predicts a sustainable but fragile co-existence of multiple species.

A more robust way to maintain high diversity is to include space [13–18] e.g. in combination with hypercycles [19] or predator-prey cycles [20–23]. Studying ecosystems in marine hard-substrate environments Jackson & Buss [24] suggested that non-transitive allelopathic relationships between species could maintain species diversity on a longer timescale than pure hierarchical predation relationships. This was confirmed in a model on two-dimensional lattice where sessile species compete for space [25, 26]. As in the ”Buss“ model [25, 26] we consider a 2-dimensional lattice where the competition for resources is a zero sum game about available space. In our model, however, the focus is on the dynamic balance between an introduction of new species and exclusion of older species. With this complementary model we, for the first time show that: a) There is a sharp transition from multiple to single species as the number of interactions is increased. b) Both cycles and chain-like (hierarchical) relationships lead to spatial fragmentation of species population thus creating isolated niches for new species and increased diversity.

The model is inspired by the spatial dynamics of lichen communities. Lichens have existed for as long as ∼ 600 million years [27] and are organisms consisting of fungi and algae living in a symbiosis [28]. Communities of lichens are formed by a combination of slow local growth and a reproductive strategy where fungal spores or propagules containing the intact symbiosis are dispersed (e.g. by winds or water flows) over lengthscales much larger than the size of the communities [29]. Because of this long distance dispersal, we assume that new species come from far away, and are completely unrelated to any of the species that already colonize our system.

Fig. 1 shows a crustose lichen community in an alpine environment. These lichens grow about 0.1mm/year and typically covers a rock surface that has recently been exposed on a timescale ∼ 100 → 1000 years [30]. When a crustose lichen meets another, a contact boundary is formed, and if they are competitively equal the boundary remains stable over time. The bulging boundaries between various species seen in Fig. 1 suggest a mini-ecosystem with complex interactions. The interaction of these species may be represented by a directed network with directions from superior to inferior species. This network does not necessarily have any particular species as the most fit one in agreement with the fact that any sizable rock typically hosts multiple lichen species.

## Results

Our model considers a multi-species ecosystem competing on a two dimensional lattice of sites which at any given time can be occupied by one species only. The species on one site can invade a neighbor site, provided that it is occupied by a competitively inferior species. The emerging ecosystem can be characterized by a directed network of possible species interactions. Interactions which are materialized only when organisms of the respective species are neighbors somewhere in the system. The aim of our model is to study ecosystem diversity as we change the number of potential interactions between species, parametrized by γ. In addition to this, new species are introduced with a rate α. Each time step of our model consists of two possible events: (i) Select a random site i and one of its nearest neighbors j. If the species s(i) at site i can invade the species s(j) at site j, i.e. Γ(s(i), s(j)) = 1, then site j is updated by setting s(j) = s(i). Here Γ is the matrix that represents the possible interactions. These interactions remain fixed once they are introduced. (i) With probability α × γ/N a new random species s is introduced at a random point j and assigned random interactions Γ(s, u) and Γ(u, s) with all existing species u in the system. Each of these interactions are assigned value 1 with probability γ, or otherwise set to 0 (we do allow for the case Γ(u, s) = Γ(s, u) = 1). The introduced species s is assumed to be able to invade the previous species at the site j, s(j): Γ(s, s(j)) = 1.

In Fig. 2, we show snapshots of a model ecosystem of size N = L × L = 200 × 200 using open boundary conditions. The snapshots represent a typical steady state behavior using γ = 0.1 and α = 0.1. The different colors represent different species. Parts of the lattice are inactive (e.g. the orange colored species in the upper right corner), while other parts are exposed to active cycles, like the 4 mutually invading species grey → dark brown → brown → red → grey. This cycle is also illustrated in terms of a network in the lower panels where the thickness of green links represents the number of active invasion events per time unit.

With time the diversity (number of species in the system) D reaches a steady state where there is a balance between new species eliminating more than one of its prey species, and an increase in D when a new species only invades a fraction of an existing species. An increase in diversity requires that the population of existing species are fragmented into spatially separate regions. Two locally interacting species cannot increase spatial heterogeneity: One species will eventually be replaced by the other. To generate spatially separated regions, one needs simultaneous dynamics of at least 3 species locally. Such multiple interactions may be both chain-like or cyclic. For example a species A that emerges inside the territory of B, can result in a fragmentation of B if a third species C can invade A but not B. Whether B invades C or not, distinguishes a cycle from a chain-like relationship, but both can generate spatial fragmentation. However the cyclic relationship tends to live longer.

In the network Γ a link is called active if it represents an interaction between two species which are physically in contact. For low γ only a small fraction of the links Γ(s, u) = 1 are active since the species s and u are often physically separated by other species which neither can invade. For larger γ we observe fewer and more widespread species which are often in contact leading to a larger fraction of active links, see Fig. 3.

The central feature of the model is the ability to sustain high diversity, even in the limit α → 0. Fig. 4 quantifies this for a N = 200 × 200 system, which shows the finite diversity D in the limit of α → 0 for γ < 0.055. In this figure we also show D(γ) obtained from a quasistatic simulation where new species are only introduced when there is no ongoing population dynamics. For γ > $\gamma_{c}$ ≈ 0.055 the quasistatic diversity is D(γ) = 1.

The quasistatic version of the model reflects the biologically interesting limit where evolution rates are much slower than any dynamics associated to the species populations. The quasistatic simulation has two metastable states when γ < $\gamma_{c}$, one with a high but finite diversity and one absorbing state where D = 1. In the simulations, the state of high diversity is reached by starting from a state with large diversity. In simulating the quasistatic dynamics there will occasionally be long periods where several species compete dynamically for the same area. To shorten these periods we eliminate all outgoing links from one randomly chosen active species. If this does not stop the dynamics, the elimination procedure is repeated until the system eventually freezes. After the system has frozen, the eliminated links are re-introduced and the quasistatic simulation is continued by introducing a new random species. The obtained steady state diversity (black line in Fig. 4) is close to the full simulation at α = 0.01.

Importantly, the obtained large diversity at γ < $\gamma_{c}$ is also found in an extended model where species sometimes compete for free space created by occasional death of individuals, i.e. the system behaviour is robust to external pertubations of the type considered in [26]. In the quasistatic limit the sharp transition to high diversity is maintained even when 10% of the systems sites are killed prior to the introduction of each new species. For finite α, occasional death will soften the transition.

In the insert of Fig. 4, we examine a variant of the quasistatic model where two new species are introduced simultaneously instead of only one. Thereby the absorbing state at D = 1 disappears and the previously metastable state at high diversity becomes a true steady state. One sees that the obtained steady state diversity for this variant of the model is close to that obtained from the α → 0 limit of the standard model. Also, the insert in Fig. 4 shows that the behavior of the one- and two-species introduction models have a similar high diversity state when N → ∞. Both models exhibit a phase transition to the high diversity state for γ < $\gamma_{c}$ ≈ 0.055. The transition is maintained also when we allow the invasion rates to vary between the species, as well as when we implement the model on a triangular lattice. Close to the critical point, the finite but small α simulations exhibits pronounced bi-stability with rare transitions between a high and low diversity state. Transitions from low to high diversity are created through a state with many disconnected patches of a few species, whereas the breakdown of high diversity occurs when each species is only represented by a few patches.

For real sessile ecosystems, γ can be estimated from fraction of active boundaries, which have been reported down to a value of 2.5% for sponge-corals competition [31]. The low γ is also consistent with observations on crustose lichens [30].

## Discussion

The model results in an interesting interplay between an interaction network [32, 33] and spatial dynamics. That is, the spatial configuration of the species limits the number of active links in the network and as a results most links are passive and have no influence on the dynamics. This is in contrast to the common network assumption that all links represent ongoing interactions. In biological systems most links are transient. To explore further the influence of the space-network coupling, we have compared our spatial model with an equivalent random-neighbor version where lattice sites interact at random and independent of their spatial locations. The non-spatial model is a noisy version of average population equations of the form

$$
\frac{dX_i}{dt} = \sum_j \Gamma(i,j)X_iX_j - \sum_j \Gamma(j,i)X_iX_j,
$$

with the additional rule that species with near zero population are removed. By comparing the dashed and solid lines in Fig. 4, we see that the behavior of the random-neighbor model differs from our model, (i) D is independent of γ and in general much lower than for the spatial model. In the quasistatic limit then D = 1 for all γ values. (ii) At moderate α and γ the random neighbor model can develop sustained oscillatory states, resembling the hypercycles of Eigen and Shuster [1]. The size and duration of these cycles decrease with the rate of introduction of new species α. In the spatial model these cycles emerge in restricted spatial regions, cause redistribution of species boundaries and give rise to sub-populations separated by neutral species.

The process that increases diversity in the model speaks to an interesting interplay between a sympatric and an allopatric speciation process. Although we have not implemented genealogical species relationships, the algorithm might be seen as a minimal evolution model, where new species are related to the species in the region where they originate. In this framework, the spatial division of an existing species population by a neutral species and the subsequent invasion of one fragment by a new species then amount to an allopatric speciation event. Overall, our model can create two species in a region previously covered by one species.

The main feature of our spatial model is the emergence of a robust ecological system with multiple co-existing species. The species diversity depends crucially on both the limitations imposed by the interaction network, and on the spatial positioning of the individual species. If either space or network constraint is absent the ecosystem complexity cannot be maintained. The co-existence is dynamic in the sense that it includes both extinction as well as occasional re-emergence of new species. In [25] it was found that non-transitive relationships indeed prolonged co-existence of many species. However in that model a substantial fraction of the lattice was often cleared by external disturbances, and any of the initially present 10 species were repeatedly introduced on empty lattice positions. In contrast by allowing introduction of species with new random interactions, our model demonstrated a self-organization towards an ecosystem with substantial diversity provided that the interaction probability γ < $\gamma_{c}$ ∼ 0.055. A system diversity which in fact remains even when we artificially cut cyclic relationships in the quasistatic limit, demonstrating that steady state species diversity crucially depends on chain-like spatial interactions that involves at least 3 species.

In an evolutionary perspective, an interesting aspect of our model is the fact that it does not rely on any ultimate fitness landscape. At any point in space and time, the ”fitness” of a species is dictated by its neighborhood [34, 35]. Our model supplements these studies by proposing a self-organized allopatric speciation mechanism which suggest a minimal sustainable diversity for ecosystems of mutually exclusive species in 2-dimensions.

## Acknowledgment

This study was supported by the Danish National Research Foundation through the Center for Models of Life.

## Figures

### Figure 1

![Figure 1](assets/ecosystems_high_diversity/figure_1.png)

**Figure 1.** Photograph of a crustose lichen community on a rock in an alpine environment (at 1300m altitude, Jotunheimen, Norway).

### Figure 2

![Figure 2](assets/ecosystems_high_diversity/figure_2.png)

**Figure 2.** Visualization of six successive snapshots of a simulation with γ = 0.1 and α = 0.1 for a system of size L = 200. The networks below the snapshots illustrate the actual (green) and potential (grey) links between the species in the system. The size of a node represents the current population size of the corresponding species whereas the thickness of green links quantifies the number of active invasions sites.

### Figure 3

![Figure 3](assets/ecosystems_high_diversity/figure_3.png)

**Figure 3.** Active interactions between species in the system. The number of interactions are measured in units of the maximum potential interactions (sum of all outgoing links in the network) and shown as function of γ and α. The computations were carried out on a system of size L = 200.

### Figure 4

![Figure 4](assets/ecosystems_high_diversity/figure_4.png)

**Figure 4.** Diversity D as function of γ for different α values. The shaded areas mark the difference between the spatial model (upper bounding lines) and the random neighbor version (dashed lower lines). For the non-spatial version the diversity D ≈ 11 × $\alpha^{0.85}$ + 1 for L = 200 is independent of γ. The black line shows the diversity achieved from the quasistatic limit. The inset shows simulations in the quasistatic limit where two species are simultaneously introduced whenever a configuration freezes. The black line is identical to the quasistatic limit shown in the main panel. The inset also shows that D/Area acts as an order parameter for the system.

## References

[1] M. Eigen and P.Schuster, The Hypercycle: A principle of natural self-organization (Springer, Berlin, 1979).

[2] A. Garca-Tejedor, A. R. Castao, F. Morn, and F. Montero, J Mol Evol 26, 294 (1987).

[3] J. M. Smith, Nature 280, 445 (1979).

[4] P. Dayton, Ecol. Monogr. 41, 351 (1971).

[5] R. Paine, Oecologia 15, 93 (1974).

[6] G. Gause, J. of Exp. Biol. 9, 389 (1934).

[7] G. Hardin, Science 131, 1292 (1960).

[8] D. Tilman, D. Wedin, and J. Knops, Nature 379, 718 (1996).

[9] A. R. Ives and S. R. Carpenter, Science 317, 58 (2007).

[10] J. H. Vandermeer and D. H. Boucher, J. Theor. Biol. 74, 549 (1978).

[11] S. Jain and S. Krishna, Proc Natl Acad Sci 98, 543 (2001).

[12] D. Segr, D. Ben-Eli, and D. Lancet, Proc Natl Acad Sci 97, 4112 (2000).

[13] D. Mollison, J. Royal Stat. Soc.. B 39, pp. 283 (1977).

[14] K. Hufkens, P. Scheunders, and R. Ceulemans, Ecol. Res. 24, 977 (2009), 10.1007/s11284-009-0584-7.

[15] P. Kareiva, A. Mullen, and R. Southwood, Philosophical Transactions: Biological Sciences 330, 175 (1990).

[16] B. Kerr, C. Neuhauser, B. J. M. Bohannan, and A. M. Dean, Nature 442, 75 (2006).

[17] S. J. Schrag and J. E. Mittler, Am. Nat. 148, 348 (1996).

[18] S. Heilmann, K. Sneppen, and S. Krishna, J Virol 84, 3016 (2010).

[19] M. Boerlijst and P. Hogeweg, Physica D 48, 17 (1991).

[20] M. E. Gilpin, Am. Nat. 109, 51 (1975).

[21] M. Perc and A. Szolnoki, New J. Phys. 9, 267 (2007).

[22] R. Laird and B. Schamp, Am. Nat. 168, 182 (2006).

[23] G. Szabo and T. Czaran, Phys. Rev. E 64, 042902 (2001).

[24] J. Jackson and L. Buss, Proc. Natl Acad. Sci. 72, 5160 (1975).

[25] L. Buss and J. Jackson, Am. Nat. 113, 223 (1979).

[26] R. Karlson and L. Buss, Ecological Modeling 23, 243 (1984).

[27] X. Yuan, S. Xiao, and T. N. Taylor, Science 308, 1017 (2005).

[28] T. H. Nash, Lichen Biology (Cambridge University Press, 2008).

[29] J. Munoz, A. M. Felicisimo, F. Cabezas, A. R. Burgaz, and I. Martinez, Science 304, 1144 (2004).

[30] E. Jettestuen, A. Nermoen, G. Hestmark, E. Timdal, and J. Mathiesen, PLoS One 5, e12820 (2010).

[31] L. Aerts and R. Soest, Mar. Ecol. Prog. Ser. 148, 125 (1997).

[32] M. Rosvall, I. B. Dodd, S. Krishna, and K. Sneppen, Phys Rev E 74, 066105 (2006).

[33] R. V. Sole and J. M. Montoya, Proc Biol Sci 268, 2039 (2001).

[34] S. A. Kauffman and S. Johnsen, J Theor Biol 149, 467 (1991).

[35] K. Sneppen, P. Bak, H. Flyvbjerg, and M. H. Jensen, Proc Natl Acad Sci U S A 92, 5209 (1995).
