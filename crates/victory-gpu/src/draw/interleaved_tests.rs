use super::{WRITE_29, buffer, compile_cs, root_view, title_list};
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

/// The late miss depends on the actual middle dispatch output. The producer
/// and late dispatch mark writes through the real submission memory map.
/// The ordinary middle shader uses a different shadow for its result.
const PRODUCE: &str = r#"
ByteAddressBuffer pages : register(t0);
RWByteAddressBuffer target : register(u0);
RWByteAddressBuffer dirty : register(u1);
cbuffer input : register(b0) { uint region; uint page; }
[RootSignature("SRV(t0), UAV(u0), UAV(u1), RootConstants(num32BitConstants=2,b0)")]
[numthreads(1, 1, 1)]
void main() {
    uint entry = pages.Load(region * 4) * 4096 + page;
    uint ignored;
    dirty.InterlockedOr((entry >> 5) * 4, 1u << (entry & 31), ignored);
    target.Store(0, 29);
}
"#;

const READ: &str = r#"
cbuffer Input : register(b0) { uint value; }
RWByteAddressBuffer result : register(u0);
RWByteAddressBuffer dirty : register(u1);
cbuffer options : register(b1) { uint mark; uint bit; }
[RootSignature("CBV(b0), UAV(u0), UAV(u1), RootConstants(num32BitConstants=2,b1)")]
[numthreads(1, 1, 1)]
void main() {
    result.Store(0, value);
    if (mark != 0) {
        uint ignored;
        dirty.InterlockedOr((bit >> 5) * 4, 1u << (bit & 31), ignored);
    }
}
"#;

const FINISH: &str = r#"
ByteAddressBuffer pages : register(t0);
RWByteAddressBuffer target : register(u0);
RWByteAddressBuffer dirty : register(u1);
RWByteAddressBuffer middle : register(u2);
cbuffer input : register(b0) { uint region; uint page; uint allowMiss; }
[RootSignature("SRV(t0), UAV(u0), UAV(u1), UAV(u2), RootConstants(num32BitConstants=3,b0)")]
[numthreads(1, 1, 1)]
void main() {
    uint value = middle.Load(0);
    uint entry = pages.Load(region * 4) * 4096 + page;
    uint ignored;
    dirty.InterlockedOr((entry >> 5) * 4, 1u << (entry & 31), ignored);
    if (allowMiss != 0 && value == 29) {
        dirty.InterlockedOr(0, 1, ignored);
    }
    target.Store(16, value);
}
"#;

#[test]
#[ignore = "requires a dedicated GPU process and isolated frame withholding"]
fn a_clean_late_computed_dispatch_receives_the_staged_producer() {
    interleaved(false, false, false, false);
}

#[test]
#[ignore = "requires a dedicated GPU process and isolated frame withholding"]
fn a_late_miss_keeps_derived_ordinary_output_private() {
    interleaved(true, false, false, false);
}

#[test]
#[ignore = "requires a dedicated GPU process and isolated frame withholding"]
fn a_late_miss_keeps_derived_marked_pages_private() {
    interleaved(true, true, false, false);
}

#[test]
#[ignore = "requires a dedicated GPU process and isolated frame withholding"]
fn a_readback_sink_disables_private_staging() {
    interleaved(true, false, true, false);
}

#[test]
#[ignore = "requires a dedicated GPU process, DXIL validator and isolated frame withholding"]
fn translated_root_cbv_writes_stay_private_on_late_miss() {
    interleaved(true, true, false, true);
}

#[test]
#[ignore = "requires a dedicated GPU process, DXIL validator and isolated frame withholding"]
fn translated_root_cbv_writes_publish_without_late_miss() {
    interleaved(false, true, false, true);
}

fn translated_middle() -> super::HostPipeline {
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
    };
    use victory_shader::{emit, io, ir, resource, structure};
    let words = [
        0xbe82_0390u32,
        0xbe83_0380, // CBV extent and flags
        0xbe86_0390,
        0xbe87_0380, // UAV extent and flags
        0xf420_0200,
        0xfa00_0000, // s_buffer_load_dword s8, s[0:3], 0
        0xbf8c_0070, // wait for scalar load
        0x7e06_0208, // v_mov_b32 v3, s8
        0xe070_0000,
        0x8001_0300, // buffer_store_dword v3, off, s[4:7], 0
        0xbf81_0000,
    ];
    let mut native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 6,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    for (slot, user_data, kind) in [(0, 0, ResourceType::Cbv), (1, 4, ResourceType::Uav)] {
        native.layout.push(RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data,
            slot,
        });
        native.used.push(UsedResource {
            kind,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data,
            slot,
        });
    }
    let p = ir::build(&native.code, false, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let resources = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let stage = io::raise(&p, &native).unwrap();
    let container = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    let mut shader = emit::emit_with(
        &container,
        &native,
        &p,
        &st,
        &stage,
        &resources,
        emit::Options { view_marks: true },
    )
    .unwrap();
    assert_eq!(shader.root_cbvs, [0]);
    assert!(shader.view_marks);
    let mut validator =
        emit::validator::Validator::load(emit::validator::Validator::default_path().as_deref())
            .unwrap();
    validator.validate(&mut shader.dxil).unwrap();
    super::HostPipeline::new(
        &[crate::pipeline::synthetic_compute_shader(
            shader,
            native.used,
        )],
        None,
    )
    .unwrap()
}

fn interleaved(allow_miss: bool, marked: bool, readback_sink: bool, translated: bool) {
    let host = crate::host::host().expect("interleaved regression needs a GPU");
    let (produce_root, produce) = compile_cs(PRODUCE);
    let (read_root, read) = compile_cs(READ);
    let (finish_root, finish) = compile_cs(FINISH);
    let (unrelated_root, unrelated_pso) = compile_cs(WRITE_29);
    let page = crate::watch::PAGE;
    let title = (0x55_2000_0000usize..0x55_2000_0000usize + page * 64)
        .step_by(page)
        .find_map(|at| {
            // SAFETY: a private committed test page at an available address.
            let data = unsafe {
                VirtualAlloc(
                    at as *const _,
                    page,
                    MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            };
            (!data.is_null()).then_some(data as usize)
        })
        .expect("low title page");
    let address = title + 16;
    // SAFETY: private allocated title page.
    unsafe { (address as *mut u32).write(11) };
    crate::shadow::register(title, title as u64, page as u64);
    let target = crate::shadow::find(address as u64, 4).unwrap();
    let ordinary = crate::watch::tests::pool(1);
    crate::shadow::register(ordinary, ordinary as u64, page as u64);
    let ordinary_gpu = crate::shadow::find(ordinary as u64, page as u64).unwrap();
    let unrelated = buffer(4);
    let (unrelated_data, _) = crate::resource::buffer_bytes(unrelated).unwrap();
    let unrelated_shadow = crate::shadow::find(unrelated_data as u64, 4).unwrap();

    let (title_list, allocator) = title_list();
    let prefix = crate::list::host_list(title_list).unwrap();
    let map = crate::memory::bind(prefix, crate::memory::OURS).unwrap();
    let args = [
        (title >> 28) as u32,
        ((title >> 16) & 4095) as u32,
        u32::from(allow_miss),
    ];
    // SAFETY: the open prefix and its arguments/heap are alive.
    unsafe {
        prefix.list.SetComputeRootSignature(&produce_root);
        prefix.list.SetPipelineState(&produce);
        prefix.list.SetComputeRootShaderResourceView(0, map.table);
        prefix
            .list
            .SetComputeRootUnorderedAccessView(1, target.gpu(address as u64));
        prefix.list.SetComputeRootUnorderedAccessView(2, map.dirty);
        prefix
            .list
            .SetComputeRoot32BitConstants(3, 2, args.as_ptr().cast(), 0);
        prefix.list.Dispatch(1, 1, 1);
    }
    prefix.note_buffer_dispatch(true);
    unrelated_shadow.refresh(prefix, unrelated_data as u64, 4);
    unrelated_shadow.gpu_writes(prefix, unrelated_data as u64, 4, false);
    // SAFETY: an independent bounded dispatch writing a different shadow.
    unsafe {
        prefix.list.SetComputeRootSignature(&unrelated_root);
        prefix.list.SetPipelineState(&unrelated_pso);
        prefix
            .list
            .SetComputeRootUnorderedAccessView(0, unrelated_shadow.gpu(unrelated_data as u64));
        prefix.list.Dispatch(1, 1, 1);
    }
    prefix.note_buffer_dispatch(false);
    let middle = crate::list::cpu_access_list(title_list, &[(address, 4)]).unwrap();
    assert!(!std::ptr::eq(prefix, middle));
    let mut state = super::State::default();
    let mut sampler = std::ptr::null();
    if translated {
        // The translated root signature declares both directly indexed
        // heap types. Supply a title sampler heap even though this shader
        // only reads a CBV and writes a buffer.
        let desc = [1u32, 4, 1, 0, 0, 0, 0, 0];
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                desc.as_ptr().cast(),
                &iid,
                &mut sampler,
            ),
            crate::object::S_OK
        );
        state.set_heaps(&[sampler]);
        super::set(state.roots(true), 0, super::Root::View(address as u64));
        super::set(state.roots(true), 1, super::Root::View(ordinary as u64));
        state.dispatch_bound(middle, &translated_middle(), 1, 1, 1);
    } else {
        let cbv = root_view(middle, address as u64, false, false).unwrap();
        assert_ne!(cbv, target.gpu(address as u64));
        ordinary_gpu.refresh(middle, ordinary as u64, page as u64);
        ordinary_gpu.gpu_writes(middle, ordinary as u64, 4, marked);
        let dirty = crate::shadow::view_writes(false).unwrap();
        let options = [
            u32::from(marked),
            ordinary_gpu.mark_entry(ordinary as u64).unwrap().0,
        ];
        // SAFETY: the ordinary middle recording holds both resources and is open.
        unsafe {
            middle.list.SetComputeRootSignature(&read_root);
            middle.list.SetPipelineState(&read);
            middle.list.SetComputeRootConstantBufferView(0, cbv);
            middle
                .list
                .SetComputeRootUnorderedAccessView(1, ordinary_gpu.gpu(ordinary as u64));
            middle.list.SetComputeRootUnorderedAccessView(2, dirty);
            middle
                .list
                .SetComputeRoot32BitConstants(3, 2, options.as_ptr().cast(), 0);
            middle.list.Dispatch(1, 1, 1);
        }
        middle.note_buffer_dispatch(false);
    }
    let all_uav = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: orders the bounded buffer writes before the late consumer.
    unsafe { middle.list.ResourceBarrier(&[all_uav]) };
    middle.close();

    let late = host.new_list().unwrap();
    let map = crate::memory::bind(&late, crate::memory::OURS).unwrap();
    // SAFETY: the late list and all its GPU resources live until its fence.
    unsafe {
        late.list.SetComputeRootSignature(&finish_root);
        late.list.SetPipelineState(&finish);
        late.list.SetComputeRootShaderResourceView(0, map.table);
        late.list
            .SetComputeRootUnorderedAccessView(1, target.gpu(address as u64));
        late.list.SetComputeRootUnorderedAccessView(2, map.dirty);
        late.list
            .SetComputeRootUnorderedAccessView(3, ordinary_gpu.gpu(ordinary as u64));
        late.list
            .SetComputeRoot32BitConstants(4, 3, args.as_ptr().cast(), 0);
        late.list.Dispatch(1, 1, 1);
    }
    let readback = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    if readback_sink {
        let dst = readback.clone();
        let src = ordinary_gpu.clone();
        // A texture readback has this same unclassified completion
        // interface. No private staging is permissible in this group.
        late.defer_after(move || {
            dst.store(
                src.read_host(ordinary as u64, 1)[0],
                std::sync::atomic::Ordering::Relaxed,
            )
        });
    }
    late.note_buffer_dispatch(true);
    late.close();
    host.execute(&[prefix, middle, &late]);

    // SAFETY: submission is complete and both guest allocations are live.
    let published = unsafe {
        [
            (address as *const u32).read(),
            (ordinary as *const u32).read(),
            ((address + 16) as *const u32).read(),
        ]
    };
    let gpu = [
        target.read_host(address as u64, 1)[0],
        ordinary_gpu.read_host(ordinary as u64, 1)[0],
        target.read_host((address + 16) as u64, 1)[0],
    ];
    let missed = crate::memory::withholding();

    // SAFETY: an independent title output must still be delivered on a miss.
    assert_eq!(unsafe { unrelated_data.cast::<u32>().read() }, 29);
    if missed {
        // The miss must also invalidate the private result for future
        // submissions that refresh this original native shadow.
        ordinary_gpu.update(ordinary, page);
        assert_eq!(ordinary_gpu.read_host(ordinary as u64, 1), [0]);
    }
    drop(state);
    for resource in [unrelated, title_list, allocator, sampler]
        .into_iter()
        .filter(|r| !r.is_null())
    {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(resource, 0x10)(
            resource,
        );
    }
    crate::shadow::unregister(title, title as u64);
    crate::shadow::unregister(ordinary, ordinary as u64);
    crate::memory::prepare(crate::memory::OURS, |_, _, _| {});
    drop(target);
    crate::watch::tests::free(title);
    drop(ordinary_gpu);
    crate::watch::tests::free(ordinary);

    println!(
        "interleaved case allow_miss={allow_miss}: published={published:?}, GPU={gpu:?}, missed={missed}"
    );
    if readback_sink {
        assert!(!missed);
        assert_eq!(gpu, [29, 11, 11]);
        assert_eq!(readback.load(std::sync::atomic::Ordering::Relaxed), 11);
    } else {
        assert_eq!(
            gpu, [29; 3],
            "each native stage must consume its actual producer"
        );
        assert_eq!(missed, allow_miss);
        let expected = if allow_miss { [11, 0, 0] } else { [29; 3] };
        assert_eq!(published, expected);
    }
}
