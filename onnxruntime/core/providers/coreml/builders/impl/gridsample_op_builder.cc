// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class GridSampleOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status GridSampleOpBuilder::AddToModelBuilderImpl([[maybe_unused]] ModelBuilder& model_builder,
                                                  [[maybe_unused]] const Node& node,
                                                  [[maybe_unused]] const logging::Logger& logger) const {
#if defined(COREML_ENABLE_MLPROGRAM)
  using namespace CoreML::Specification::MILSpec;  // NOLINT
  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.resample

  const auto input_defs = node.InputDefs();
  const auto output_defs = node.OutputDefs();

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Failed to get input shape");

  NodeAttrHelper helper(node);
  const auto& mode = helper.Get("mode", "linear");
  const bool align_corners = helper.Get("align_corners", 0);
  const std::string coordinates_mode = "normalized_minus_one_to_one";
  std::string padding_mode = helper.Get("padding_mode", "zeros");
  if (padding_mode == "zeros") {
    padding_mode = "constant";
  }

  // grid is normalized values for H, W, 2.
  // need to multiply by input H and W to denormalize, and cast to int32 for coreml
  std::vector<float> h_w_float = {static_cast<float>(input_shape[2]), static_cast<float>(input_shape[3])};
  const int32_t float_elem_type = static_cast<int32_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  const int32_t int32_elem_type = static_cast<int32_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  auto denormalize = model_builder.CreateOperation(node, "mul");
  AddOperationInput(*denormalize, "x", input_defs[1]->Name());
  AddOperationInput(*denormalize, "y", model_builder.AddConstant(denormalize->type(), "h_w", h_w_float));
  const auto& denormalize_output = model_builder.GetUniqueName(node, "denorm");
  AddIntermediateOperationOutput(*denormalize, denormalize_output, float_elem_type, input_shape);

  auto cast = model_builder.CreateOperation(node, "cast");
  AddOperationInput(*cast, "x", denormalize_output);
  AddOperationInput(*cast, "dtype", model_builder.AddScalarConstant(cast->type(), "to", std::string("int32")));
  const auto& cast_output = model_builder.GetUniqueName(node, "to_int32");
  AddIntermediateOperationOutput(*cast, cast_output, int32_elem_type, input_shape);

  auto op = model_builder.CreateOperation(node, "resample");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationInput(*op, "coordinates", cast_output);
  AddOperationInput(*op, "sampling_mode", model_builder.AddScalarConstant(op->type(), "sampling_mode", mode));
  AddOperationInput(*op, "padding_mode", model_builder.AddScalarConstant(op->type(), "padding_mode", padding_mode));
  AddOperationInput(*op, "pading_value", model_builder.AddScalarConstant(op->type(), "padding_value", 0.0f));
  AddOperationInput(*op, "coordinates_mode",
                    model_builder.AddScalarConstant(op->type(), "coordinates_mode", coordinates_mode));
  AddOperationInput(*op, "align_corners", model_builder.AddScalarConstant(op->type(), "align_corners", align_corners));

  AddOperationOutput(*op, *output_defs[0]);
#endif
  return Status::OK();
}

bool GridSampleOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "GridSample is not supported.";
    return false;
  }

  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  const auto input_rank = input_shape.size();
  if (input_rank < 4) {
    LOGS(logger, VERBOSE) << "GridSample only supports 4D input not " << input_rank << "D";
  }

  if (input_shape[2] < 0 || input_shape[3] < 0) {
    // we need H and W to denormalize the 'grid' input
    LOGS(logger, VERBOSE) << "GridSample requires H and W dimensions to be known";
    return false;
  }

  NodeAttrHelper helper(node);

  const auto& mode = helper.Get("mode", "linear");
  if (mode == "cubic") {
    LOGS(logger, VERBOSE) << "GridSample does not support cubic interpolation";
    return false;
  }

  return true;
}

void CreateGridSampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GridSampleOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
