import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


class DecodingError(Exception):
    """Raised if no path ends in the zero state"""
    pass


class TrellisPath:
    def __init__(self, last_state=0):
        self._path_metric = 0
        self._path = [last_state]
        self._last_state = last_state
        self._bits_input = [None]
        self._len = 1

    def add_2_path(self, state, branch_metric, bits_input):
        self._last_state = state
        self._path.append(state)
        self._path_metric += branch_metric
        self._bits_input.append(bits_input)
        self._len += 1

    def get_path(self):
        return self._path.copy()

    def path_metric(self):
        return self._path_metric

    def last_state(self):
        return self._last_state

    def input_bits(self):
        return self._bits_input

    def __repr__(self):
        return " -> ".join(map(str, self._path))

    def __len__(self):
        return self._len

    @classmethod
    def duplicate_path(cls, path):
        if not isinstance(path, TrellisPath):
            raise ValueError("must receive a valid path")
        new_path = TrellisPath()
        new_path._path = path._path.copy()
        new_path._path_metric = path._path_metric
        new_path._last_state = path._last_state
        new_path._bits_input = path._bits_input.copy()
        new_path._len = path._len
        return new_path


class ConvolutionalCode:
    """The code assumes zero state termination, and k=1"""
    def __init__(self, generators: tuple):
        """
        :param generators: each element in the tuple represents a single generator polynomial. The convention
        we use is: 1+D =b011 = 3 (and not 1+D=6).
        """
        self.n = len(generators)
        self.k = 1
        self.rate = self.k / self.n
        self.constraint_length = math.floor(math.log(max(generators), 2))
        self.number_of_states = 2 ** self.constraint_length
        self.state_space = tuple(range(self.number_of_states))
        self.generators = generators

        self._build_fsm(generators)

    def _build_fsm(self, generators: tuple):
        possible_inputs = tuple(range(2 ** self.k))
        self.next_states = {}
        self.out_bits = {}
        for current_state in self.state_space:
            self.next_states[current_state] = {}
            self.out_bits[current_state] = {}

            for current_input in possible_inputs:

                new_state = (current_input << (self.constraint_length - 1)) + (current_state >> self.k)
                self.next_states[current_state][current_input] = new_state

                tmp = []
                for fwd in generators:
                    bit_reversed_fwd = int('{:0{width}b}'.format(fwd, width=self.constraint_length+1)[::-1], 2)
                    lsr = (current_input << self.constraint_length) + current_state
                    generator_masked_sum_arg = bit_reversed_fwd & (lsr)  # mask input and state with fwd
                    tmp.append(bin(generator_masked_sum_arg).count("1") % 2)  # sum bit mod 2 (XOR)

                self.out_bits[current_state][current_input] = tuple(tmp)

    def encode(self, data: bytes) -> (list,list,list):
        """
        encode input data bytes
        :param data: date to be encoded
        :return: encoded data bits, as a list of integers of value 0 or 1
        :rtype: bytes
        """
        
        input_bits = [0] * (self.constraint_length + len(data) * 8)
        coded_bits = [0] * int(len(input_bits)/self.rate)
        states_route = [0] * int(len(input_bits)+1)

        for byte_idx, byt in enumerate(data):
            bits = '{:08b}'.format(byt)
            input_bits[8*byte_idx:8*byte_idx+8] = bits
            
        
        current_state = 0
        
        for bit_idx, bit in enumerate(input_bits):
            outputs = list(self.out_bits[current_state][int(bit)])
            coded_bits[self.n*bit_idx: self.n*(bit_idx+1)] = outputs
            states_route[bit_idx] = current_state;
            current_state = self.next_states[current_state][int(bit)]
        states_route[bit_idx+1] = current_state;
        return (coded_bits,states_route,input_bits)

    def decode(self, data: list) -> (bytes, int, TrellisPath):
        """
        decode data bytes
        :param data: coded data to be decoded, list of ints representing each received bit.
        :return: return a tuple of decoded data, and the amount of corrected errors.
        :rtype: (bytes, int)

        The function assumes initial and final state of encoder was at the zero state
        """
        received_codewords = [tuple(data[i: i+self.n]) for i in range(0, len(data), self.n)]
        surviving_paths = [TrellisPath()]

        for codeword in received_codewords:  # iterate over time (received codewords)
            # obtain branch metrics
            possible_transitions = []
            # find branch metrics
            for path in surviving_paths:
                for possible_input in range(2**self.k):
                    # for each path and possible input find possible output, and branch metric
                    last_state = path.last_state()
                    next_state = self.next_states[last_state][possible_input]
                    possible_output = self.out_bits[last_state][possible_input]
                    branch_metric = sum(tuple(possible_output[i]^codeword[i] for i in range(len(codeword))))
                    possible_transitions.append([next_state, branch_metric + path.path_metric(), branch_metric, path,
                                                 possible_input])

            # select survivors by inspecting paths entering a state
            new_paths = []
            for state in self.state_space:
                entering_paths = tuple(filter(lambda x: x[0] == state, possible_transitions))
                # initially there may be less paths than states, since we assume initialization at zero state
                if len(entering_paths):
                    selected = min(entering_paths, key=lambda x: x[1])
                    selected_path: TrellisPath = selected[3]
                    new_path = TrellisPath.duplicate_path(selected_path)
                    new_path.add_2_path(state, selected[2], selected[4])
                    new_paths.append(new_path)
            surviving_paths = new_paths

        # choose ML path
        chosen_path = None
        for path in surviving_paths:
            if path.last_state() == 0:  # as a result of zero tailing
                chosen_path = path
                break
        if chosen_path is None:
            raise DecodingError

        decoded_bits = chosen_path.input_bits()[1:-int(self.constraint_length)]
        mapped = "".join(map(str, decoded_bits))
        decoded_bytes = bytes([int(mapped[i:i + 8], 2) for i in range(0, len(mapped), 8)])
        chosen_path.path_metric()
        return decoded_bytes, chosen_path.path_metric(), chosen_path

    def print_generators(self):
        for generator_idx, generator_p in enumerate(self.generators):
            binary_rep = '{:0{width}b}'.format(generator_p, width=self.constraint_length+1)[::-1]
            function_rep = ""
            for bit_idx, bit in enumerate(binary_rep):
                if bit == "1":
                    if bit_idx == 0:
                        function_rep = "1 + "
                    else:
                        function_rep = function_rep + "x^" + str(bit_idx) + " + "
            if len(function_rep) > 3:
                function_rep = function_rep[:-3]
            print("generator no. " + str(generator_idx) + ": "+ function_rep)

    def print_fsm(self):
        possible_inputs = tuple(range(2 ** self.k))
        for state in self.state_space:
            for current_input in possible_inputs:
                print("current state: ", state, ", current input: ", current_input, ", new state: ",
                      self.next_states[state][current_input], ", encoder output:", self.out_bits[state][current_input])

    def plot_trellis(self, state_path: list, input_bits: list):
        fig, ax = plt.subplots(figsize=(10,5))
        size_x = len(state_path)
        X, Y = np.meshgrid(np.arange(size_x), np.arange(self.number_of_states))
        ax.scatter(X,Y,c='black')
        xc = range(size_x)
        ax.plot(xc, state_path, linewidth=0, marker="o", color="black",
             markersize=np.sqrt(200), markerfacecolor='none', markeredgewidth=3)
        line_styles=['solid', 'dashed']
        line_colors=['red','blue']
        for i in range(len(xc)-1):
            cp = ConnectionPatch((xc[i],state_path[i]), (xc[i+1], state_path[i+1]), 
                                 coordsA='data', coordsB='data', axesA=ax, axesB=ax,
                                 shrinkA=np.sqrt(300)/2, shrinkB=np.sqrt(300)/2,
                                 linewidth=2,linestyle=line_styles[int(input_bits[i])],color=line_colors[int(input_bits[i])])
            ax.add_patch(cp)
            
        ax.yaxis.get_major_locator().set_params(integer=True)
        plt.gca().invert_yaxis()
        plt.show()

def pass_channel (x_bin: list, target_snr_db: float) -> list:
    x_volts = np.array(x_bin)*2-1
    x_watts = x_volts ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise_volts = np.random.normal(0, np.sqrt(noise_avg_watts), len(x_volts))
    y_volts = x_volts + noise_volts
    binary_data = (y_volts >= 0).astype(int)
    return list(binary_data)
    

if __name__ == "__main__":
    #PARAMETERS
    SNR = 0; #dB
    to_plot  =0;
    NBytes= 10000;
    
    #input data set
    if NBytes == 0:
        input_bytes = b"\xFE\xF0\x0A\x01" #trama personalizada 
        NBytes = len(input_bytes);
    else:
        input_bytes = [0]*(NBytes*8)
        mapped = "".join(map(str, input_bytes))
        input_bytes = bytes([int(mapped[i:i + 8], 2) for i in range(0, len(mapped), 8)])    
    
    
    # example of constructing encoder A
    # and generators:
    #       g1(x) = 1 + x^2, represented in binary as b101 = 5
    #       g2(x) = 1 + x+ x^2, represented in binary as b111 = 7
    conv = ConvolutionalCode((5, 7))
    
    # encoding a byte stream
    encoded,states_list,input_bits = conv.encode(input_bytes)
    
    #plot codded trellis path
    if to_plot==1:
        conv.plot_trellis(states_list,input_bits)

    #introducing noise
    corrupted=pass_channel(encoded.copy(), SNR)
    decoded, corrected_errors, path = conv.decode(corrupted)
    
    #plot chosen path 
    if to_plot==1:
        conv.plot_trellis(path.get_path(),path.input_bits()[1:])
    
    print("SNR = ", SNR)
    print("Encoder A")    
    print("Decoded == input bytes?: ", decoded == input_bytes)
    print("Errores corregidos:", corrected_errors) #errors "corrected" by decoder
    
    #COMPUTE BER HERE:
    
    num_errorsA = sum([bin(input_bytes[i] ^ decoded[i]).count('1') for i in range(len(input_bytes))])
    BER_A = num_errorsA / (len(input_bytes) * 8)  # Divide por la longitud total de bits
    print("Bit Error Rate (BER) for encoder A:", BER_A)
    

    #-----------------------------------------------------------------------------

    # example of constructing an encoder B
    # and generators:
    #       g1(x) = 1 + x, represented in binary as b011 = 3
    #       g2(x) = 1 + x + x^2, represented in binary as b111 = 7
    #       g3(x) = 1 + x^2 + x^3, represented in binary as b1101 = 13
    conv = ConvolutionalCode((3, 7, 13))

    # encoding a byte stream
    encoded,states_list,input_bits = conv.encode(input_bytes)
    
    #plot codded trellis path
    if to_plot==1:
        conv.plot_trellis(states_list,input_bits)

    #introducing noise
    corrupted=pass_channel(encoded.copy(), SNR)
    decoded, corrected_errors, path = conv.decode(corrupted)
    
    #plot chosen path 
    if to_plot==1:
        conv.plot_trellis(path.get_path(),path.input_bits()[1:])
        
    print("\nEncoder B") 
    print("Decoded == input bytes?: ", decoded == input_bytes)
    print("Errores corregidos:", corrected_errors) #errors "corrected" by decoder
    
    #COMPUTE BER HERE:
    
    num_errorsB = sum([bin(input_bytes[i] ^ decoded[i]).count('1') for i in range(len(input_bytes))])
    BER_B = num_errorsB / (len(input_bytes) * 8)  # Divide por la longitud total de bits
    print("Bit Error Rate (BER) for encoder B:", BER_B)
    

    