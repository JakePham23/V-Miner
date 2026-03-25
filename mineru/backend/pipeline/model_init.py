import os

import torch
from loguru import logger

from .model_list import AtomicModel
from ...model.layout.doclayoutyolo import DocLayoutYOLOModel
from ...model.mfd.yolo_v8 import YOLOv8MFDModel
from ...model.mfr.unimernet.Unimernet import UnimernetModel
from ...model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from ...model.ori_cls.paddle_ori_cls import PaddleOrientationClsModel
from ...model.table.cls.paddle_table_cls import PaddleTableClsModel
# from ...model.table.rec.RapidTable import RapidTableModel
from ...model.table.rec.slanet_plus.main import RapidTableModel
from ...model.table.rec.unet_table.main import UnetTableModel
from ...utils.config_reader import get_device
from ...utils.enum_class import ModelPath
from ...utils.models_download_utils import auto_download_and_get_model_root_path

MFR_MODEL = os.getenv('MINERU_FORMULA_CH_SUPPORT', 'False')
if MFR_MODEL.lower() in ['true', '1', 'yes']:
    MFR_MODEL = "pp_formulanet_plus_m"
elif MFR_MODEL.lower() in ['false', '0', 'no']:
    MFR_MODEL = "unimernet_small"
else:
    logger.warning(f"Invalid MINERU_FORMULA_CH_SUPPORT value: {MFR_MODEL}, set to default 'False'")
    MFR_MODEL = "unimernet_small"


def img_orientation_cls_model_init():
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang="ch_lite",
        enable_merge_det_boxes=False
    )
    cls_model = PaddleOrientationClsModel(ocr_engine)
    return cls_model


def table_cls_model_init():
    table_backend = os.getenv('MINERU_TABLE_BACKEND', 'paddle').lower()
    if table_backend == 'lighton':
        logger.info("Using LightOn backend for Table Classification")
    return PaddleTableClsModel()

def wired_table_model_init(lang=None):
    table_backend = os.getenv('MINERU_TABLE_BACKEND', 'paddle').lower()
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        enable_merge_det_boxes=False
    )
    if table_backend == 'lighton' and lang and lang.startswith('vi'):
        logger.info("Using LightOn backend for Wired Table recognition (Vietnamese)")
        # LightOnOCR logic is handled in batch_analyze.py, 
        # but we initialize the default here as fallback
    table_model = UnetTableModel(ocr_engine)
    return table_model


def wireless_table_model_init(lang=None):
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        enable_merge_det_boxes=False
    )
    table_model = RapidTableModel(ocr_engine)
    return table_model


def mfd_model_init(weight, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    mfd_model = YOLOv8MFDModel(weight, device)
    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    if MFR_MODEL == "unimernet_small":
        mfr_model = UnimernetModel(weight_dir, device)
    elif MFR_MODEL == "pp_formulanet_plus_m":
        mfr_model = FormulaRecognizer(weight_dir, device)
    else:
        logger.error('MFR model name not allow')
        exit(1)
    return mfr_model


def doclayout_yolo_model_init(weight, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    model = DocLayoutYOLOModel(weight, device)
    return model

def ocr_model_init(det_db_box_thresh=0.3,
                   lang=None,
                   det_db_unclip_ratio=1.8,
                   enable_merge_det_boxes=True
                   ):
    """
    Initialize OCR model based on language.
    
    Supports multiple OCR backends:
    - vi: EasyOCR (better Vietnamese support)
    - vi-vision: Apple Vision Framework (macOS only, high accuracy)
    - vi-light-ocr: LightOnOCR via LM Studio (best for tables)
    - vi-paddle-ocr: Original PaddleOCR for Vietnamese
    - others: PaddleOCR (default)
    """
    # Route to appropriate OCR backend
    if lang == 'vi-light-ocr':
        # Hybrid: EasyOCR for text + LightOnOCR for tables (via LM Studio)
        from mineru.model.ocr.hybrid_light_ocr import HybridLightOCR
        logger.info("Using Hybrid OCR backend (EasyOCR + LightOnOCR) for Vietnamese")
        return HybridLightOCR()
    
    elif lang == 'vi-vision-light':
        # Hybrid: Vision Framework for text + LightOnOCR for tables
        import sys
        if sys.platform != 'darwin':
            logger.warning(f"vi-vision-light mode requires macOS, falling back to vi-light-ocr")
            from mineru.model.ocr.hybrid_light_ocr import HybridLightOCR
            return HybridLightOCR()
        
        from mineru.model.ocr.hybrid_vision_light_ocr import HybridVisionLightOCR
        logger.info("Using Hybrid OCR backend (Vision Framework + LightOnOCR) for Vietnamese")
        return HybridVisionLightOCR()
    
    elif lang == 'vi-hybrid':
        # Custom hybrid OCR via environment variables
        import sys
        primary_ocr = os.getenv('PRIMARY_OCR', 'easyocr').lower()
        table_ocr = os.getenv('TABLE_OCR', 'lighton').lower()
        
        logger.info(f"Using Custom Hybrid OCR: PRIMARY_OCR={primary_ocr}, TABLE_OCR={table_ocr}")
        
        if primary_ocr == 'vision':
            # Vision + LightOnOCR
            if sys.platform != 'darwin':
                logger.warning("Vision OCR requires macOS, falling back to EasyOCR")
                primary_ocr = 'easyocr'
            else:
                from mineru.model.ocr.hybrid_vision_light_ocr import HybridVisionLightOCR
                return HybridVisionLightOCR()
        
        # Default: EasyOCR + LightOnOCR
        from mineru.model.ocr.hybrid_light_ocr import HybridLightOCR
        return HybridLightOCR()
    
    elif lang == 'vi-custom':
        # Configurable hybrid OCR with flexible backend selection for ALL content types.
        # Defaults: easyocr for all - works offline, no server needed.
        # Override via env variables:
        #   MINERU_TEXT_BACKEND=paddle|easyocr|vision (macOS only)
        #   MINERU_TABLE_BACKEND=paddle|easyocr|vision|lighton (lighton needs LM Studio)
        #   MINERU_IMAGE_BACKEND=paddle|easyocr|vision|lighton (lighton needs LM Studio)
        text_backend = os.getenv('MINERU_TEXT_BACKEND', 'easyocr').lower()
        table_backend = os.getenv('MINERU_TABLE_BACKEND', 'rapidtable').lower()
        image_backend = os.getenv('MINERU_IMAGE_BACKEND', 'paddle').lower()
        
        logger.info(
            f"Using Configurable Hybrid OCR: "
            f"text={text_backend}, table={table_backend}, image={image_backend}"
        )
        
        from mineru.model.ocr.configurable_hybrid_ocr import ConfigurableHybridOCR
        return ConfigurableHybridOCR(
            text_backend=text_backend,
            table_backend=table_backend,
            image_backend=image_backend,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    
    elif lang == 'vi-vision':

        # Apple Vision Framework (macOS only)
        import sys
        if sys.platform != 'darwin':
            logger.warning(f"vi-vision mode requires macOS, falling back to EasyOCR")
            from mineru.model.ocr.easy_ocr import EasyOCR
            return EasyOCR(lang='vi')
        
        from mineru.model.ocr.vision_ocr import VisionFrameworkOCR
        logger.info("Using Apple Vision Framework for Vietnamese OCR")
        return VisionFrameworkOCR()

    
    elif lang in ['vi', 'vietnamese', 'vie']:
        # EasyOCR for Vietnamese
        from mineru.model.ocr.easy_ocr import EasyOCR
        logger.info("Using EasyOCR backend for Vietnamese")
        return EasyOCR(lang='vi')
    
    elif lang == 'vi-paddle-ocr':
        # Original PaddleOCR for Vietnamese (fallback option)
        logger.info("Using PaddleOCR backend for Vietnamese")
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            lang='vi',  # Map to PaddleOCR's Vietnamese support
            use_dilation=True,
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
        )
        return model
    
    elif lang is not None and lang != '':
        # PaddleOCR for other languages
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            lang=lang,
            use_dilation=True,
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
        )
    else:
        # Default PaddleOCR
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            use_dilation=True,
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
        )
    return model



class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):

        lang = kwargs.get('lang', None)

        if atom_model_name in [AtomicModel.WiredTable, AtomicModel.WirelessTable]:
            key = (
                atom_model_name,
                lang
            )
        elif atom_model_name in [AtomicModel.OCR]:
            key = (
                atom_model_name,
                kwargs.get('det_db_box_thresh', 0.3),
                lang,
                kwargs.get('det_db_unclip_ratio', 1.8),
                kwargs.get('enable_merge_det_boxes', True)
            )
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]

def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = doclayout_yolo_model_init(
            kwargs.get('doclayout_yolo_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('det_db_box_thresh', 0.3),
            kwargs.get('lang'),
            kwargs.get('det_db_unclip_ratio', 1.8),
            kwargs.get('enable_merge_det_boxes', True)
        )
    elif model_name == AtomicModel.WirelessTable:
        atom_model = wireless_table_model_init(
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.WiredTable:
        atom_model = wired_table_model_init(
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.TableCls:
        atom_model = table_cls_model_init()
    elif model_name == AtomicModel.ImgOrientationCls:
        atom_model = img_orientation_cls_model_init()
    else:
        logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model


class MineruPipelineModel:
    def __init__(self, **kwargs):
        self.formula_config = kwargs.get('formula_config')
        self.apply_formula = self.formula_config.get('enable', True)
        self.table_config = kwargs.get('table_config')
        self.apply_table = self.table_config.get('enable', True)
        self.lang = kwargs.get('lang', None)
        self.device = kwargs.get('device', 'cpu')
        logger.info(
            'DocAnalysis init, this may take some times......'
        )
        atom_model_manager = AtomModelSingleton()

        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(
                    os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd), ModelPath.yolo_v8_mfd)
                ),
                device=self.device,
            )

            # 初始化公式解析模型
            if MFR_MODEL == "unimernet_small":
                mfr_model_path = ModelPath.unimernet_small
            elif MFR_MODEL == "pp_formulanet_plus_m":
                mfr_model_path = ModelPath.pp_formulanet_plus_m
            else:
                logger.error('MFR model name not allow')
                exit(1)

            self.mfr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=str(os.path.join(auto_download_and_get_model_root_path(mfr_model_path), mfr_model_path)),
                device=self.device,
            )

        # 初始化layout模型
        self.layout_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            doclayout_yolo_weights=str(
                os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
            ),
            device=self.device,
        )
        # 初始化ocr
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=self.lang
        )
        # init table model
        if self.apply_table:
            self.wired_table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.WiredTable,
                lang=self.lang,
            )
            self.wireless_table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.WirelessTable,
                lang=self.lang,
            )
            self.table_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.TableCls,
            )
            self.img_orientation_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.ImgOrientationCls,
                lang=self.lang,
            )

        logger.info('DocAnalysis init done!')


class HybridModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        lang=None,
        formula_enable=None,
    ):
        key = (lang, formula_enable)
        if key not in self._models:
            self._models[key] = MineruHybridModel(
                lang=lang,
                formula_enable=formula_enable,
            )
        return self._models[key]

def ocr_det_batch_setting(device):
    # 检测torch的版本号
    import torch
    from packaging import version

    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")

    if (
            version.parse(torch.__version__) >= version.parse("2.8.0")
            or str(device).startswith('mps')
            or device_type.lower() in ["corex"]
    ):
        enable_ocr_det_batch = False
    else:
        enable_ocr_det_batch = True
    return enable_ocr_det_batch

class MineruHybridModel:
    def __init__(
            self,
            device=None,
            lang=None,
            formula_enable=True,
    ):
        if device is not None:
            self.device = device
        else:
            self.device = get_device()

        self.lang = lang

        self.enable_ocr_det_batch = ocr_det_batch_setting(self.device)

        if str(self.device).startswith('npu'):
            try:
                import torch_npu
                if torch_npu.npu.is_available():
                    torch_npu.npu.set_compile_mode(jit_compile=False)
            except Exception as e:
                raise RuntimeError(
                    "NPU is selected as device, but torch_npu is not available. "
                    "Please ensure that the torch_npu package is installed correctly."
                ) from e

        self.atom_model_manager = AtomModelSingleton()

        # 初始化OCR模型
        self.ocr_model = self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=self.lang
        )

        if formula_enable:
            # 初始化公式检测模型
            self.mfd_model = self.atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(
                    os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd), ModelPath.yolo_v8_mfd)
                ),
                device=self.device,
            )

            # 初始化公式解析模型
            if MFR_MODEL == "unimernet_small":
                mfr_model_path = ModelPath.unimernet_small
            elif MFR_MODEL == "pp_formulanet_plus_m":
                mfr_model_path = ModelPath.pp_formulanet_plus_m
            else:
                logger.error('MFR model name not allow')
                exit(1)

            self.mfr_model = self.atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=str(os.path.join(auto_download_and_get_model_root_path(mfr_model_path), mfr_model_path)),
                device=self.device,
            )