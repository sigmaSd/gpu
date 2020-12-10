use gpu::{calc, execute_gpu, execute_gpu_sync};

fn main() {
    //sync
    let spirv = calc!(b0 += b1 * b2 + b3 + 4);
    let r = execute_gpu_sync(
        vec![vec![0, 0, 0], vec![5, 4, 0], vec![1, 2, 0], vec![2, 5, 4]],
        spirv.as_binary(),
    );
    assert_eq!(r, vec!(11, 17, 8));
    // async
    let spirv = calc!(b0 += 4);
    let r = futures::executor::block_on(execute_gpu(vec![vec![0, 0, 0]], spirv.as_binary()));
    assert_eq!(r, vec!(4, 4, 4));
}
