###### VERISON2
import inspect
import os
import threading

from dotenv import load_dotenv

from scripts.auto_tests_docs.docs import DOCUMENTATION_WRITER_SOP
from swarms import OpenAIChat

##########
from zeta.nn.modules.simple_mamba import MambaBlock, Mamba
from zeta.nn.modules.laser import Laser
from zeta.nn.modules.fused_gelu_dense import FusedDenseGELUDense
from zeta.nn.modules.fused_dropout_layernom import FusedDropoutLayerNorm
from zeta.nn.modules.conv_mlp import Conv2DFeedforward
from zeta.nn.modules.ws_conv2d import WSConv2d
from zeta.nn.modules.stoch_depth import StochDepth
from zeta.nn.modules.nfn_stem import NFNStem
from zeta.nn.modules.film import Film
from zeta.nn.modules.proj_then_softmax import FusedProjSoftmax
from zeta.nn.modules.top_n_gating import TopNGating
from zeta.nn.modules.moe_router import MoERouter
from zeta.nn.modules.perceiver_layer import PerceiverLayer
from zeta.nn.modules.u_mamba import UMambaBlock
from zeta.nn.modules.vit_denoiser import (
    VisionAttention,
    VitTransformerBlock,
)
from zeta.nn.modules.v_layernorm import VLayerNorm
from zeta.nn.modules.parallel_wrapper import Parallel
from zeta.nn.modules.v_pool import DepthWiseConv2d, Pool
from zeta.nn.modules.moe import MixtureOfExperts
from zeta.nn.modules.flex_conv import FlexiConv
from zeta.nn.modules.mm_layernorm import MMLayerNorm
from zeta.nn.modules.fusion_ffn import MMFusionFFN
from zeta.nn.modules.norm_utils import PostNorm
from zeta.nn.modules.mm_mamba_block import MultiModalMambaBlock
from zeta.nn.modules.p_scan import PScan
from zeta.nn.modules.ssm import SSM
from zeta.nn.modules.film_conditioning import FilmConditioning



####################
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    openai_api_key=api_key,
    max_tokens=2000,
)


def process_documentation(cls):
    """
    Process the documentation for a given class using OpenAI model and save it in a Markdown file.
    """
    doc = inspect.getdoc(cls)
    source = inspect.getsource(cls)
    input_content = (
        "Class Name:"
        f" {cls.__name__}\n\nDocumentation:\n{doc}\n\nSource"
        f" Code:\n{source}"
    )

    # Process with OpenAI model (assuming the model's __call__ method takes this input and returns processed content)
    processed_content = model(
        DOCUMENTATION_WRITER_SOP(input_content, "zeta.nn.modules")
    )

    # doc_content = f"# {cls.__name__}\n\n{processed_content}\n"
    doc_content = f"{processed_content}\n"

    # Create the directory if it doesn't exist
    dir_path = "docs/zeta/nn/modules"
    os.makedirs(dir_path, exist_ok=True)

    # Write the processed documentation to a Markdown file
    file_path = os.path.join(dir_path, f"{cls.__name__.lower()}.md")
    with open(file_path, "w") as file:
        file.write(doc_content)

    print(f"Documentation generated for {cls.__name__}.")


def main():
    classes = [
        MambaBlock,
        Mamba,
        Laser,
        FusedDenseGELUDense,
        FusedDropoutLayerNorm,
        Conv2DFeedforward,
        WSConv2d,
        StochDepth,
        NFNStem,
        Film,
        FusedProjSoftmax,
        TopNGating,
        MoERouter,
        PerceiverLayer,
        UMambaBlock,
        VisionAttention,
        VitTransformerBlock,
        VLayerNorm,
        Parallel,
        DepthWiseConv2d,
        Pool,
        MixtureOfExperts,
        FlexiConv,
        MMLayerNorm,
        MMFusionFFN,
        PostNorm,
        MultiModalMambaBlock,
        PScan,
        SSM,
        FilmConditioning,
    ]
    
    threads = []
    for cls in classes:
        thread = threading.Thread(target=process_documentation, args=(cls,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Documentation generated in 'docs/zeta/nn/modules' directory.")


if __name__ == "__main__":
    main()
