#include "../baseline_checkpoint_choice.hpp"
#include <cassert>

int main() {
    using namespace qw3::baseline;
    // Regression: replay starts at 30208, captures a newer P at 38304, ends
    // unaligned at 38352. Do not replace that P with the old replay origin.
    auto pick = choose_latest_aligned_checkpoint({38304, 4, true}, {30208, 4, true}, 38352, 4, 32);
    assert(pick.position == 38304 && !pick.postselection);
    // A newly selected window can invalidate the older-generation P.
    pick = choose_latest_aligned_checkpoint({38336, 3, true}, {30208, 4, true}, 38352, 4, 32);
    assert(pick.position == 30208 && pick.postselection);
    // Neither a prompt-end partial block nor a state beyond the committed end
    // can replace an existing replay-safe origin.
    pick = choose_latest_aligned_checkpoint({38352, 4, true}, {30208, 4, true}, 38352, 4, 32);
    assert(pick.position == 30208);
    pick = choose_latest_aligned_checkpoint({38368, 4, true}, {30208, 4, true}, 38352, 4, 32);
    assert(pick.position == 30208);
    pick = choose_latest_aligned_checkpoint({38336, 4, false}, {30208, 3, true}, 38352, 4, 32);
    assert(pick.position == 0);
}
