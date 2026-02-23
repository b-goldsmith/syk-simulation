from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils import PauliMask, PauliSum, pauli_sum_to_numpy 
from ..jw_transform import SYK_hamil
from .SYK_qdrift import SYK_qdrift


import numpy as np
from pytest import mark
from scipy.linalg import expm


def compare_SYK_qdrift_eigen(number_qubits,coefs, J, t, eps):
    hamil = SYK_hamil(number_qubits, J, coefs)
    mat = pauli_sum_to_numpy(hamil)
    eigsys = np.linalg.eigh(mat)

    qpu = QPU(num_qubits = number_qubits)
    qubits = Qubits(num_qubits=number_qubits,qpu=qpu)
    qubits.push_state(eigsys[1][0])

    SYK_qdrift(qubits = qubits, time = t, epsilon = eps, J=J, coefs =coefs)
    vec = qpu.pull_state()

    assert np.allclose(eigsys[1][0], vec, rtol=0.001, atol=0.01)

def compare_SYK_qdrift_unitary(number_qubits,coefs, J, t, eps):
    hamil = SYK_hamil(number_qubits, J, coefs = coefs)
    exact_mat = expm(-1j * pauli_sum_to_numpy(hamil) * t)

    qpu = QPU(num_qubits = number_qubits, filters=">>unitary>>")
    qubits = Qubits(num_qubits=number_qubits,qpu=qpu)

    SYK_qdrift(qubits = qubits, time = t, epsilon = eps, J=J, coefs =coefs)
    ufilter = qpu.get_filter_by_name(">>unitary>>")
    matrix = ufilter.get()
    qpu.reset(number_qubits)

    assert np.allclose(exact_mat, matrix, rtol=0.001, atol=0.01)

@mark.parametrize(
    "number_qubits,coefs, J, t, eps",
    [
        (3, [-0.020696423042134694, -0.00576101254879936,
            0.026987022420862185,-0.06345957735033439,
            0.009756390613472332, 0.00975570653954919,
            0.06580053397947463, 0.031976447048037863,
            -0.01956143274728967, 0.022606668482748525,
            0.01930907053385259, 0.0194054063987607,
            -0.01008176131525142, 0.0797200101940749,
            0.0718715763547097], 1, 2, 0.0001),
        (4, [-0.013442721091468693,
            -0.003741887414085888, 0.017528585241725153,
            0.041218204574109865, -0.006336961590383312,
            0.006336517271336682, -0.04273870050660465,
            -0.020769311599776533, 0.012705523270180258,
            -0.014683461900744927, -0.012541609204336425,
            -0.0126041811840659, -0.006548296060424209,
            0.05177966551351747, 0.04668195819990868,
            -0.015217352642308366, -0.027410546248533275,
            0.008504567909343902, -0.024574122395289007,
            -0.03822159010047353, -0.039665283340982946,
            0.006110250368556525, 0.0018275356478655459,
            -0.038558378851770485, -0.014732789650631068,
            0.0030019306419469495, 0.031149677426263917,
            -0.010167625876213249, -0.01625526137392154,
            -0.007894193670191235, -0.016284162869241565,
            0.05012874883313937, -0.0003652793594887043,
            -0.028625141698007728, 0.022260774676093747,
            0.033040050466369444, -0.005652536849995711,
            0.053035128447415106, 0.0359450893530663,
            -0.005327713477595666, 0.01998533806630746,
            0.0046377901535434194,0.0031298234516329,
            0.008148857798553141,0.04001367511600385,0.01948135535115366,-0.012466402425600546,
            0.028609209462835662,-0.009299442749101404,0.04771367382613097,-0.00877077970172472,0.010421594918367904,0.018319739020173056,-0.01655397515714936,-0.027902243055347477,-0.025203507537313312,-0.02271199045037069,-0.008368305395365427,
             -0.008965079591884067,-0.026401464457065083,0.01296803321290837,
             -0.005024543444733996,-0.02994106851826303,-0.032373291394658846,
            0.021989625107006826,-0.03670432244911455,0.0019488310818179732,
            -0.027158905722122127,-0.009787062019214898,0.017459065499100646], 
            1, 2, 0.0001)
    ],
)

def test_SYK_qdrift_eigen(number_qubits, coefs, J, t, eps):
    compare_SYK_qdrift_eigen(number_qubits = number_qubits,coefs = coefs, J=J, t=t, eps=eps)
    compare_SYK_qdrift_unitary(number_qubits = number_qubits,coefs = coefs, J=J, t=t, eps=eps)



