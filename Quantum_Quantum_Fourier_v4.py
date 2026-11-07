# Quantum_4_Fourier



import csv
import itertools
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)


TITLE = "Quantum Fourier"
N, K, DIM = 39, 7, 64
REGULARIZATIONS = (0.001, 0.01)
MAX_ITERATIONS = 200

DIFFERENCES = (
    np.arange(DIM)[:, None] - np.arange(N)[None, :]
) % DIM

# QFT -> kompleksni spektralni filter -> inverse QFT.
# Uce se spektralne amplitude i relativne faze.
#
# Numericki operator racuna se FFT postupkom.
# Qiskit ga nezavisno proverava QFT kolom.
#
# Jedan pomocni kubit ostvaruje unitarnu dilataciju filtera.
# Grana pomocnog kubita 0 dobija se iz tacnih amplituda.
# Nema slucajnog uzorkovanja.


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            try:
                values = tuple(int(value) for value in row)
            except ValueError as error:
                raise ValueError(
                    f"{path.name}: necelobrojan red {line}."
                ) from error

            check(
                len(values) == K
                and len(set(values)) == K
                and min(values) >= 1
                and max(values) <= N
                and values == tuple(sorted(values)),
                f"{path.name}: nelegalan red {line}.",
            )
            rows.append(values)

    check(len(rows) >= 5, f"{path.name}: nedovoljno kola.")
    return np.asarray(rows, dtype=int) - 1


def initial():
    return np.concatenate((
        np.zeros(DIM),
        0.2 * np.sin(np.arange(1, DIM)),
    ))


def transition(theta):
    magnitudes = expit(theta[:DIM])
    phases = np.concatenate(([0.0], theta[DIM:]))
    phase_factors = np.exp(1j * phases)
    coefficients = magnitudes * phase_factors

    success_probability = float(
        np.mean(magnitudes**2)
    )
    check(
        success_probability > 0.0,
        "Filter nema uspesnu granu.",
    )

    impulse = (
        np.fft.fft(coefficients)
        / DIM
        / np.sqrt(success_probability)
    )
    amplitudes = impulse[DIFFERENCES]

    return np.abs(amplitudes)**2, (
        amplitudes,
        magnitudes,
        phase_factors,
        coefficients,
        success_probability,
    )


def log_amplitudes(amplitudes):
    magnitudes = np.abs(amplitudes)
    result = np.full(magnitudes.shape, -np.inf)
    np.log(
        magnitudes,
        out=result,
        where=magnitudes > 0.0,
    )
    return 2.0 * result


def log_transition(theta):
    _, auxiliary = transition(theta)
    return log_amplitudes(auxiliary[0])


def data_loss(probabilities, counts):
    check(
        probabilities.shape == (DIM, N)
        and np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0)
        and np.allclose(
            probabilities.sum(axis=0),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Neispravna uslovna raspodela.",
    )

    total = counts.sum()
    previous_counts = counts.sum(axis=0)

    loss = (
        np.sum(probabilities**2 * previous_counts)
        - 2.0 * np.sum(counts * probabilities)
    ) / total + 1.0

    derivative = 2.0 * (
        probabilities * previous_counts - counts
    ) / total

    return float(loss), derivative


def objective(theta, counts, weight):
    probabilities, auxiliary = transition(theta)
    (
        amplitudes,
        magnitudes,
        phase_factors,
        coefficients,
        success,
    ) = auxiliary

    loss, derivative = data_loss(probabilities, counts)

    # Sve kolone operatora su pomeranja istog impulsnog odziva.
    difference_gradient = np.bincount(
        DIFFERENCES.ravel(),
        weights=derivative.ravel(),
        minlength=DIM,
    )
    impulse = amplitudes[:, 0]

    inner = np.fft.fft(
        difference_gradient * impulse.conj()
    ) / DIM

    magnitude_derivative = (
        magnitudes * (1.0 - magnitudes)
    )
    scale_term = float(np.dot(
        difference_gradient,
        np.abs(impulse)**2,
    ))

    magnitude_gradient = (
        2.0
        * np.real(
            magnitude_derivative * phase_factors * inner
        )
        / np.sqrt(success)
        - 2.0
        * magnitudes
        * magnitude_derivative
        * scale_term
        / (DIM * success)
    )

    phase_gradient = (
        2.0
        * np.real(1j * coefficients * inner)
        / np.sqrt(success)
    )[1:]

    gradient = np.concatenate((
        magnitude_gradient,
        phase_gradient,
    ))

    return (
        loss + weight * np.dot(theta, theta),
        gradient + 2.0 * weight * theta,
    )


def joint(log_weights):
    logs = np.asarray(log_weights, dtype=float).copy()
    check(
        logs.ndim == 2
        and np.all(np.isfinite(logs) | np.isneginf(logs)),
        "Neispravni logaritmi raspodele.",
    )

    slots, states = logs.shape
    check(1 <= slots <= states, "Neispravne dimenzije.")

    indices = np.arange(states)
    increasing = indices[:, None] < indices[None, :]
    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )
    logs[~allowed] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[0] = logs[0]
    values = logs[0].copy()
    parents = np.zeros((slots, states), dtype=int)

    for slot in range(1, slots):
        forward[slot] = logs[slot] + logsumexp(
            np.where(
                increasing,
                forward[slot - 1, :, None],
                -np.inf,
            ),
            axis=0,
        )

        candidates = np.where(
            increasing,
            values[:, None],
            -np.inf,
        )
        parents[slot] = candidates.argmax(axis=0)
        values = logs[slot] + candidates[
            parents[slot], np.arange(states)
        ]

    log_partition = float(logsumexp(forward[-1]))
    check(np.isfinite(log_partition), "Nema legalne mase.")

    path = np.zeros(slots, dtype=int)
    path[-1] = int(values.argmax())

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            np.where(
                increasing,
                (
                    logs[slot + 1]
                    + backward[slot + 1]
                )[None, :],
                -np.inf,
            ),
            axis=1,
        )

    marginals = np.exp(
        forward + backward - log_partition
    )
    mode_log_mass = float(
        logs[np.arange(slots), path].sum()
    )

    check(
        np.all(np.diff(path) > 0)
        and mode_log_mass <= log_partition + 1e-8
        and log_partition <= 1e-8,
        "Neispravan maksimum ili particiona funkcija.",
    )
    check(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Marginalne raspodele nisu normalizovane.",
    )

    return path, marginals, log_partition, logs


def counts_for(data, end, slot):
    counts = np.zeros((DIM, N))
    np.add.at(
        counts,
        (
            data[1:end + 1, slot],
            data[:end, slot],
        ),
        1.0,
    )
    return counts


def fit(data, end, weight):
    parameters = []

    for slot in range(K):
        counts = counts_for(data, end, slot)
        result = minimize(
            objective,
            initial(),
            args=(counts, weight),
            jac=True,
            method="CG",
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": 1e-6,
            },
        )

        loss, gradient = objective(result.x, counts, weight)
        check(
            np.isfinite(loss)
            and np.all(np.isfinite(result.x))
            and np.all(np.isfinite(gradient)),
            "Obuka nije vratila konacne parametre i gradijent.",
        )

        print(
            "Registar:",
            slot + 1,
            "status:",
            result.message,
            flush=True,
        )
        parameters.append(result.x)

    return parameters


def evaluate(parameters, data, times):
    matrices = [
        log_transition(theta)
        for theta in parameters
    ]
    losses, hits = [], []

    for time in times:
        log_weights = np.array([
            matrices[slot][:N, data[time - 1, slot]]
            for slot in range(K)
        ])
        path, _, log_partition, logs = joint(log_weights)

        loss = log_partition - logs[
            np.arange(K), data[time]
        ].sum()

        check(
            not np.isnan(loss) and loss >= -1e-8,
            "Neispravan NLL.",
        )
        losses.append(loss)
        hits.append(len(set(path) & set(data[time])))

    return (
        float(np.mean(losses)),
        np.asarray(hits, dtype=int),
    )


def verify_quantum(theta, previous):
    probabilities, auxiliary = transition(theta)
    (
        _,
        magnitudes,
        _,
        coefficients,
        success,
    ) = auxiliary

    diagonal = np.diag(coefficients)
    complement = np.diag(
        np.sqrt(1.0 - magnitudes**2)
    )
    dilation = np.block([
        [diagonal, -complement],
        [complement, diagonal.conj().T],
    ])

    check(
        np.allclose(
            dilation.conj().T @ dilation,
            np.eye(2 * DIM),
            atol=2e-10,
            rtol=0,
        ),
        "Spektralna dilatacija nije unitarna.",
    )

    circuit = QuantumCircuit(7)
    circuit.append(
        QFT(6, do_swaps=True),
        range(6),
    )
    circuit.unitary(dilation, range(7))
    circuit.append(
        QFT(6, do_swaps=True).inverse(),
        range(6),
    )

    output = Statevector.from_int(
        previous, 2 * DIM
    ).evolve(circuit)

    branch = np.asarray(output.data[:DIM])
    branch_mass = float(np.vdot(branch, branch).real)

    check(
        np.isclose(
            branch_mass,
            success,
            atol=2e-10,
            rtol=0,
        ),
        "Neispravna masa grane kvantnog filtera.",
    )
    check(
        np.allclose(
            np.abs(branch)**2 / branch_mass,
            probabilities[:, previous],
            atol=2e-9,
            rtol=0,
        ),
        "QFT filter i FFT nisu usaglaseni.",
    )


def self_check(data):
    counts = counts_for(
        data, min(8, len(data) - 1), 0
    )
    theta = initial() + 0.03 * np.sin(
        np.arange(len(initial())) + 1
    )

    verify_quantum(theta, int(data[0, 0]))
    _, gradient = objective(theta, counts, 0.001)

    for index in range(len(theta)):
        offset = np.zeros_like(theta)
        offset[index] = 1e-6

        numerical = (
            objective(theta + offset, counts, 0.001)[0]
            - objective(theta - offset, counts, 0.001)[0]
        ) / 2e-6

        check(
            np.isclose(
                numerical,
                gradient[index],
                atol=2e-5,
                rtol=2e-4,
            ),
            f"Gradijent nije potvrdjen: {index}.",
        )

    identity_parameters = initial()
    identity_parameters[DIM:] = 0.0
    identity_probabilities, _ = transition(
        identity_parameters
    )

    check(
        np.allclose(
            identity_probabilities,
            np.eye(DIM)[:, :N],
            atol=1e-14,
            rtol=0,
        ),
        "Identicni spektralni filter nije identitet.",
    )

    tiny_log = log_amplitudes(
        np.array([1e-180 + 0j])
    )[0]
    check(
        np.isfinite(tiny_log)
        and np.isclose(
            tiny_log,
            2.0 * np.log(1e-180),
            atol=1e-10,
            rtol=0,
        ),
        "Provera veoma male amplitude nije prosla.",
    )

    toy = softmax(
        np.sin(np.arange(15).reshape(3, 5)),
        axis=1,
    )
    path, _, log_partition, logs = joint(np.log(toy))
    scores = [
        sum(
            logs[j, combination[j]]
            for j in range(3)
        )
        for combination in itertools.combinations(range(5), 3)
    ]

    check(
        np.isclose(
            log_partition,
            logsumexp(scores),
            atol=1e-10,
            rtol=0,
        )
        and np.isclose(
            logs[np.arange(3), path].sum(),
            max(scores),
            atol=1e-10,
            rtol=0,
        ),
        "Provera globalne raspodele nije prosla.",
    )

    print("Matematicke i pocetne Qiskit provere su prosle.")


def main():
    print(TITLE)
    print("Naucene spektralne amplitude i faze.")
    print("Sest sistemskih i jedan pomocni kubit za filter.")

    datasets = [
        (path, load(path))
        for path in FILES
    ]
    self_check(datasets[0][1])
    results = []

    for path, data in datasets:
        total = len(data) - 1
        train_end = int(0.70 * total)
        validation_end = int(0.85 * total)

        check(
            0 < train_end < validation_end < total,
            "Neispravna hronoloska podela.",
        )

        print()
        print(path.name, "redova:", len(data))
        best_weight, best_loss = None, np.inf

        for weight in REGULARIZATIONS:
            parameters = fit(data, train_end, weight)
            loss, _ = evaluate(
                parameters,
                data,
                range(train_end + 1, validation_end + 1),
            )

            print(
                "Regularizacija:",
                weight,
                "validacioni NLL:",
                loss,
            )

            if np.isfinite(loss) and loss < best_loss:
                best_weight, best_loss = weight, loss

        check(
            best_weight is not None,
            "Nema konacnog validacionog NLL.",
        )

        test_parameters = fit(
            data, validation_end, best_weight
        )
        test_loss, hits = evaluate(
            test_parameters,
            data,
            range(validation_end + 1, len(data)),
        )

        print("Test kola:", len(hits))
        print("Test NLL:", test_loss)
        print(
            "Test srednji broj pogodaka:",
            float(hits.mean()),
        )

        parameters = fit(data, total, best_weight)
        log_weights = []

        for slot, theta in enumerate(parameters):
            previous = int(data[-1, slot])
            verify_quantum(theta, previous)
            log_weights.append(
                log_transition(theta)[:N, previous]
            )

            _, auxiliary = transition(theta)
            print(
                "Registar:",
                slot + 1,
                "masa filtera:",
                auxiliary[-1],
            )

        log_weights = np.asarray(log_weights)
        prediction, marginals, log_partition, _ = joint(
            log_weights
        )

        print("Log legalne mase:", log_partition)

        if np.array_equal(prediction, data[-1]):
            print("PROVERA: NEXT ponavlja poslednje kolo.")

        run = 1
        for difference in np.diff(prediction):
            run = run + 1 if difference == 1 else 1
            if run >= 5:
                print(
                    "PROVERA: najmanje pet uzastopnih brojeva."
                )
                break

        results.append((
            path.name,
            prediction + 1,
            marginals,
            hits,
            np.exp(log_weights),
        ))

    for first, second in itertools.combinations(results, 2):
        if np.array_equal(first[1], second[1]):
            print("ISTI NEXT:", first[0], second[0])
            print(
                "Razlika raspodela:",
                float(np.max(np.abs(first[4] - second[4]))),
            )

    figure, axes = plt.subplots(
        3, 2, figsize=(12, 10)
    )

    for row, result in enumerate(results):
        name, prediction, marginals, hits, _ = result

        print(name)
        print("NEXT:", " ".join(map(str, prediction)))

        axes[row, 0].imshow(
            marginals,
            aspect="auto",
        )
        axes[row, 0].set_title(name)
        axes[row, 0].set_xlabel(
            "Broj - indeks 0 predstavlja broj 1"
        )
        axes[row, 1].bar(
            np.arange(8),
            np.bincount(hits, minlength=8),
        )
        axes[row, 1].set_xlabel(
            "Pogodaka na testu"
        )

    figure.suptitle(TITLE)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



"""
Quantum Fourier
Naucene spektralne amplitude i faze.
Sest sistemskih i jedan pomocni kubit za filter.
Matematicke i pocetne Qiskit provere su prosle.

loto7_4716_k89.csv redova: 4716
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Regularizacija: 0.001 validacioni NLL: 19.537916974491182
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Regularizacija: 0.01 validacioni NLL: 27.45244017854451
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Test kola: 708
Test NLL: 19.31317384232328
Test srednji broj pogodaka: 1.2429378531073447
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Registar: 1 masa filtera: 0.24841541232520348
Registar: 2 masa filtera: 0.24788841967965553
Registar: 3 masa filtera: 0.248697406210384
Registar: 4 masa filtera: 0.24851883700382496
Registar: 5 masa filtera: 0.24857741539160516
Registar: 6 masa filtera: 0.24851822239091687
Registar: 7 masa filtera: 0.24819317047789463
Log legalne mase: -4.176994708391016
PROVERA: NEXT ponavlja poslednje kolo.

loto7_4716_k89_loto_2980.csv redova: 2979
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Maximum number of iterations has been exceeded.
Regularizacija: 0.001 validacioni NLL: 19.566191041863735
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 27.36925424934838
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Optimization terminated successfully.
Test kola: 447
Test NLL: 19.59409009015499
Test srednji broj pogodaka: 1.2527964205816555
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Registar: 1 masa filtera: 0.24816806229168623
Registar: 2 masa filtera: 0.24823714181019174
Registar: 3 masa filtera: 0.24802167998927738
Registar: 4 masa filtera: 0.24866425603340095
Registar: 5 masa filtera: 0.2476716481854152
Registar: 6 masa filtera: 0.24793133694222616
Registar: 7 masa filtera: 0.24782845861724223
Log legalne mase: -4.952687430412002
PROVERA: NEXT ponavlja poslednje kolo.

loto7_4716_k89_loto_plus_1736.csv redova: 1735
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Regularizacija: 0.001 validacioni NLL: 19.38668115724426
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 27.20875120551749
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Test kola: 261
Test NLL: 19.480638737161502
Test srednji broj pogodaka: 1.3371647509578544
Registar: 1 status: Maximum number of iterations has been exceeded.
Registar: 2 status: Maximum number of iterations has been exceeded.
Registar: 3 status: Maximum number of iterations has been exceeded.
Registar: 4 status: Maximum number of iterations has been exceeded.
Registar: 5 status: Maximum number of iterations has been exceeded.
Registar: 6 status: Maximum number of iterations has been exceeded.
Registar: 7 status: Maximum number of iterations has been exceeded.
Registar: 1 masa filtera: 0.24795809098364546
Registar: 2 masa filtera: 0.24756211247028725
Registar: 3 masa filtera: 0.24857487733617556
Registar: 4 masa filtera: 0.2480634872230853
Registar: 5 masa filtera: 0.24836338776859007
Registar: 6 masa filtera: 0.24858804966446513
Registar: 7 masa filtera: 0.24772025121067254
Log legalne mase: -4.760340140874306
PROVERA: NEXT ponavlja poslednje kolo.


loto7_4716_k89.csv
NEXT: 6 12 17 25 29 31 34

loto7_4716_k89_loto_2980.csv
NEXT: 12 14 17 18 26 33 34

loto7_4716_k89_loto_plus_1736.csv
NEXT: 9 15 17 21 24 26 39
"""



"""
1. 
Ising Model	
Spinovi, međusobne interakcije, Hamiltonijan i temperaturna dinamika. 
Koristim tačan zbir nad 64 spinska stanja, uz očuvanje Hamiltonijana i temperaturne raspodele.

2. 
ODE - Ordinary Differential Equation	
Evolucija stanja preko zadate diferencijalne jednačine i početnog uslova, uz solve_ivp.	
Diferencijalna dinamika. Desnu stranu jednačine naučiti iz prethodnih kola i proveriti evoluciju raspodele. 
Otvoreni kvantni model, jer evolucija raspodele može biti neunitarna. 

3. 
Partial Differential Equations	
PDE — Heat/Diffusion
Osnovna ideja je difuzija kroz jednačinu toplote, rešavana konačnim razlikama i vremenskim koracima. 
Za Loto treba sačuvati difuzioni operator, vremensku evoluciju, početne i granične uslove, uz učenje parametara iz istorije. 
Posebno treba proveriti očuvanje ukupne verovatnoće. 
Otvoreni kvantni model, jer evolucija raspodele može biti neunitarna. 

4. 
Fourier Transform	
DFT/FFT, kompleksni koeficijenti, spektralne amplitude, faze i inverzna transformacija.	
Poseban spektralni model koji uči vremensku evoluciju komponenti raspodele. 
Fourier uči spektralne amplitude i faze. 
U Fourier kodu FFT računanje filtera, jer množenje gustih Fourier matrica može ostaviti grešku poništavanja i kod identičnog filtera. 

5, 
Quantum Mechanics	
Kompleksna talasna funkcija, Schrödingerova evolucija, potencijal, interferencija i Bornove verovatnoće.	
Kvantni dinamički model sa početnim amplitudama i parametrima naučenim iz istorije. 
Schrödingerov model uci Hermitski Hamiltonijan.



Uključene optimizacija i vizuelizacija u svaki odgovarajući model. 

Plotting and Data Visualization	
Prikaz podataka, histogrami, dijagrami i toplotne mape.	
Dijagnostika unutar prediktivnih fajlova. 
Vizuelizacija ulazi u dijagnostiku — prikaz naučene raspodele, promene kroz vreme, konvergencije i rezultata odvojenog testa.

Optimization	
Optimization ulazi kao stvarni postupak obuke. 
Gradijentni spust, promenljiv korak, konjugovane gradijente i Nelder-Mead. 
Te postupke odabrati prema ciljnoj funkciji i hronološkoj validaciji. 
Poređenje više determinističkih početnih tačaka čuva istraživanje različitih lokalnih minimuma. 
Takvo poređenje samo po sebi ne potvrđuje globalni optimum.





Fajl	                Osnovna ideja 	                                                                                Šta treba učiti iz istorije
Ising	                Spinovi, Hamiltonijan, međusobne interakcije i temperaturna raspodela.	                        Jačine interakcija, lokalna polja i njihova zavisnost od prethodnih kola.
ODE	                    Početno stanje, diferencijalna jednačina i integracija vremenske evolucije.	                    Parametri jednačine koja opisuje prelaz raspodele u sledeće kolo.
PDE — Heat/Diffusion	Difuzioni operator, konačne razlike, vremenski koraci i granični uslovi.	                    Parametri difuzije i eventualni izvorni član, uz očuvanje nenegativnosti i ukupne verovatnoće.
Fourier	                Kompleksni spektralni koeficijenti, amplitude, faze i inverzna transformacija.	                Evolucija spektralnih komponenti kroz kola.
Quantum Mechanics	    Schrödingerova evolucija, kompleksne amplitude, Hamiltonijan, interferencija i Bornovo pravilo.	Početne amplitude i parametri Hamiltonijana koji opisuju istorijske prelaze.


Za kvantnu realizaciju posebno treba očuvati razliku između difuzije i Schrödingerove evolucije. Difuzija je neunitarna. 
Njeno preslikavanje na Qiskit zahteva odgovarajuću konstrukciju. Sama zamena difuzionog operatora unitarnim kolom promenila bi metod.

Svih pet fajlova obrađuje ista tri CSV-a zasebno, sa jednom NEXT kombinacijom po CSV-u. Razlikovaće se po naučenom modelu i postupku evolucije. 



Svaki model koristi prethodno kolo kao početno stanje. 
Obuka obuhvata sve raspoložive istorijske prelaze.

- Ising koristi tačan zbir nad 64 spinska stanja.
- ODE koristi naučeni generator prenosa verovatnoće.
- PDE koristi konačne razlike, periodične granice i izvorni član koji čuva ukupnu verovatnoću.
- Fourier uči spektralne amplitude i faze. Njegov kvantni filter koristi jedan pomoćni kubit.
- Schrödingerov model uči Hermitski Hamiltonijan i čuva unitarnu evoluciju.


Optimizacija koristi konjugovane gradijente. 
Svaki fajl ima hronološku validaciju, odvojeni test, završnu obuku i grafikone.



Statička i matematička provera: 

Kod	            Nalaz

Ising	        Logaritam računati iz energije i particione funkcije. 
                Znak Hamiltonijana, Gibbsova normalizacija, gradijenti polja i interakcija, logaritamski izlaz.

ODE	            Izveo sam gradijent generatora i proverio konstrukciju prenosa. 
                Smer prenosa, zbir kolona generatora, Fréchet gradijent i konstrukcija skočnih operatora.

PDE	            Proverio sam FTCS korak, gradijent izvora i kvantni kanal. 
                Stabilnost FTCS koraka, očuvanje verovatnoće, gradijent izvora i kvantni kanal.

Fourier	        Filter  računa FFT. Time se izbegava poništavanje pri množenju gustih Fourier matrica. Logaritmi se računaju iz amplituda.
                Smer FFT/QFT, normalizacija filtera, gradijenti amplituda i faza, raspored pomoćnog kubita. 

Schrödinger     Evolucija koristi direktni expm, a gradijent odgovarajući Fréchet izvod. Logaritmi se računaju iz amplituda.
                Hermitičnost, matrična eksponencijala, kompleksni adjoint gradijent i logaritmi amplituda.


Provereni su i svi pozivi joint() — Ising, Fourier i Schrödinger verzije prosleđuju logaritme. ODE i PDE prosleđuju verovatnoće svojoj odgovarajućoj funkciji.
Hronološke granice treninga, validacije i testa pravilno su indeksirane. Završna obuka obuhvata sve istorijske prelaze.
Konvergencija optimizatora i prolazak ugrađenih provera.
"""
