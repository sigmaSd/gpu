use gpu::{calc, execute_gpu};

fn main() {
    let spirv = calc!(b0 += b1 * b2 + b3 + 4);
    let r = futures::executor::block_on(execute_gpu(
        vec![vec![0, 0, 0], vec![5, 4, 0], vec![1, 2, 0], vec![2, 5, 4]],
        spirv.as_binary(),
    ));
    assert_eq!(r, vec!(11, 17, 8));
}
