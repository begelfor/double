import time

import sys
import termios
import tty

def edit_distance(str1, str2):
    """
    Computes the Levenshtein (edit) distance between two strings.
    Returns the minimum number of single-character edits (insertions, deletions, substitutions)
    required to change str1 into str2.
    """
    m, n = len(str1), len(str2)
    # Create matrix of size (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]

def get_single_key_press():
    """
    Waits for a single key press and returns the character.
    Works on Unix-like systems (Linux, macOS).
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        # Restore old terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    seq = 'fj'
    N = 10

    s = [get_single_key_press()]
    start_time = time.time()
    s += [get_single_key_press() for _ in range(N*len(seq)-1)]
    t = time.time() - start_time

    print(f'{N*len(seq)} keys. {edit_distance(s, N*seq)} errors. {N*len(seq)/t:.2f} keys/s')
    print(''.join(s))
    print(seq*N)

        


if __name__ == '__main__':
    main()