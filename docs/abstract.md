# Implementacja algorytmu uczenia ze wzmocnieniem dla wybranej gry strategicznej

## Streszczenie

Niniejsza praca przedstawia rozwój algorytmu uczenia ze wzmocnieniem dostosowanego do strategicznej gry planszowej Blokus. Aby osiągnąć ten cel, zastosowano dwa główne algorytmy: Proksymalną Optymalizację Polityki (PPO) i AlphaZero.

Badanie rozpoczęło się od wdrożenia algorytmu PPO, wybranego ze względu na jego sukces w grach typu Atari i złożonych środowiskach, takich jak Dota 2. Wstępne testy w środowisku CartPole potwierdziły funkcjonalność implementacji PPO. Jednak pomimo kilku prób, algorytm nie działał skutecznie w środowisku Blokus i nie był w stanie przewyższyć strategii losowych ruchów.

Następnie skupiono się na AlphaZero, znanym z nadludzkiej wydajności w grach takich jak szachy, Go i Shogi. Eksperymenty rozpoczęto od uproszczonej wersji Blokusa, przechodząc do pełnej planszy 20x20 z czterema graczami. Zdolność adaptacyjna AlphaZero została zademonstrowana poprzez iteracyjny trening i rozgrywkę, wykazując znaczną poprawę strategiczną w stosunku do modeli bazowych.

Dalsze udoskonalenie algorytmu zostało ułatwione dzięki optymalizacji hiperparametrów przy użyciu frameworka Optuna. Architektura sieci neuronowej została przeskalowana, aby dopasować ją do złożoności środowiska, wykorzystując do tego głębokie splotowe sieci neuronowe. Ocena w porównaniu z klasycznym przeciwnikiem opartym na tylko na Przeszukiwaniu Drzewa Monte Carlo podkreśliła zdolność AlphaZero do optymalizacji strategii przy ograniczonych próbkach szkoleniowych, wskazując na potencjał znacznych ulepszeń przy zwiększonych zasobach obliczeniowych.

Eksperymenty przeprowadzone z PPO i AlphaZero w Blokusie podkreślają potencjalne zastosowania algorytmów uczenia ze wzmocnieniem w strategicznych grach planszowych. W szczególności AlphaZero wykazał obiecujące wyniki, sugerując, że przy większej mocy obliczeniowej jego wydajność może zostać znacznie zwiększona. Teza kończy się spostrzeżeniami na temat szerszego zastosowania tych ustaleń w różnych dziedzinach, w których strategiczne podejmowanie decyzji i zdolność adaptacji mają kluczowe znaczenie.

## Słowa kluczowe

Uczenie ze wzmocnieniem, AlphaZero, Proksymalna Optymalizacja Polityki, Blokus, Optuna, Splotowa sieć neuronowa, Przeszukiwanie Drzewa Monte Carlo

## Abstract

This thesis presents the development of a reinforcement learning algorithm tailored for the strategic board game Blokus. Two principal algorithms, Proximal Policy Optimization (PPO) and AlphaZero, were employed to achieve this objective.

The investigation began with the implementation of the PPO algorithm, chosen for its success in Atari-style games and complex environments like Dota 2. Initial tests in the CartPole environment confirmed the functionality of the PPO implementation. However, despite several attempts, the algorithm did not perform effectively in the Blokus environment and was unable to surpass a random move strategy.

Subsequently, the focus shifted to AlphaZero, renowned for its superhuman performance in games such as chess, Go, and Shogi. Experiments started with a simplified version of Blokus, advancing to the full 20x20 board with four players. AlphaZero’s adaptability was demonstrated through iterative training and gameplay, showing considerable strategic improvements over the baseline models.

Further refinement of the algorithm was facilitated through hyperparameter optimization using the Optuna framework. The architecture of the neural network was scaled up to match the complexity of the environment, using deep convolutional neural networks for this purpose. Evaluation against a classic MCTS-based opponent underscored AlphaZero’s capability to optimize strategies with limited training samples, indicating the potential for significant improvements with increased computational resources.

The experiments conducted with PPO and AlphaZero in Blokus highlight the potential applications of reinforcement learning algorithms in strategic board games. AlphaZero, in particular, showed promising results, suggesting that with greater computational power, its performance can be substantially enhanced. The thesis concludes with insights into the broader applicability of these findings in various fields where strategic decision-making and adaptability are crucial.

## Keywords

Reinforcement Learning, AlphaZero, Proximal Policy Optimization, Blokus, Optuna, Convolutional Neural Network, Monte Carlo Tree Search
