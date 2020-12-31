import math

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss


@register_criterion("bidirectional_label_smoothed_cross_entropy")
class BidirectionalLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task, sentence_avg, label_smoothing)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample["net_input"]["train_mode"] = 3
        net_output = model(**sample["net_input"])
        s2t_loss, s2t_nll_loss, t2s_loss, t2s_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        src_sample_size = (
            sample["source"].size(0) if self.sentence_avg else sample["src_ntokens"]
        )
        tgt_sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["tgt_ntokens"]
        )
        logging_output = {
            "s2t_loss": s2t_loss.data,
            "s2t_nll_loss": s2t_nll_loss.data,
            "t2s_loss": t2s_loss.data,
            "t2s_nll_loss": t2s_nll_loss.data,
            "src_ntokens": sample["src_ntokens"],
            "tgt_ntokens": sample["tgt_ntokens"],
            "ntokens": sample["src_ntokens"] + sample["tgt_ntokens"],
            "nsentences": sample["nsentences"],
            "src_sample_size": src_sample_size,
            "tgt_sample_size": tgt_sample_size,
        }
        loss = s2t_loss + t2s_loss
        sample_size = src_sample_size + tgt_sample_size
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        src_lprobs, tgt_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        src_lprobs = src_lprobs.view(-1, src_lprobs.size(-1))
        tgt_lprobs = tgt_lprobs.view(-1, tgt_lprobs.size(-1))
        source = model.get_sources(sample, net_output).view(-1, 1)
        target = model.get_targets(sample, net_output).view(-1, 1)
        s2t_loss, s2t_nll_loss = label_smoothed_nll_loss(
            tgt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        t2s_loss, t2s_nll_loss = label_smoothed_nll_loss(
            src_lprobs,
            source,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return s2t_loss, s2t_nll_loss, t2s_loss, t2s_nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        s2t_loss_sum = utils.item(sum(log.get("s2t_loss", 0) for log in logging_outputs))
        s2t_nll_loss_sum = utils.item(sum(log.get("s2t_nll_loss", 0) for log in logging_outputs))
        src_ntokens = utils.item(sum(log.get("src_ntokens", 0) for log in logging_outputs))
        src_sample_size = utils.item(sum(log.get("src_sample_size", 0) for log in logging_outputs))
        t2s_loss_sum = utils.item(sum(log.get("t2s_loss", 0) for log in logging_outputs))
        t2s_nll_loss_sum = utils.item(sum(log.get("t2s_nll_loss", 0) for log in logging_outputs))
        tgt_ntokens = utils.item(sum(log.get("tgt_ntokens", 0) for log in logging_outputs))
        tgt_sample_size = utils.item(sum(log.get("tgt_sample_size", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", s2t_loss_sum / tgt_sample_size / math.log(2) + t2s_loss_sum / src_sample_size / math.log(2), round=3
        )
        metrics.log_scalar(
            "s2t_loss", s2t_loss_sum / tgt_sample_size / math.log(2), tgt_sample_size, round=3
        )
        metrics.log_scalar(
            "t2s_loss", t2s_loss_sum / src_sample_size / math.log(2), src_sample_size, round=3
        )
        metrics.log_scalar(
            "s2t_nll_loss", s2t_nll_loss_sum / tgt_ntokens / math.log(2), tgt_ntokens, round=3
        )
        metrics.log_scalar(
            "t2s_nll_loss", t2s_nll_loss_sum / src_ntokens / math.log(2), src_ntokens, round=3
        )
        metrics.log_derived(
            "s2t_ppl", lambda meters: utils.get_perplexity(meters["s2t_nll_loss"].avg)
        )
        metrics.log_derived(
            "t2s_ppl", lambda meters: utils.get_perplexity(meters["t2s_nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("bidirectional_label_smoothed_cross_entropy_lm")
class BidirectionalLabelSmoothedCrossEntropyCriterionLM(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task, sentence_avg, label_smoothing)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample["net_input"]["train_mode"] = 2
        net_output = model(**sample["net_input"])
        s2s_loss, s2s_nll_loss, t2t_loss, t2t_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        src_sample_size = (
            sample["source"].size(0) if self.sentence_avg else sample["src_ntokens"]
        )
        tgt_sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["tgt_ntokens"]
        )
        logging_output = {
            "s2s_loss": s2s_loss.data,
            "s2s_nll_loss": s2s_nll_loss.data,
            "t2t_loss": t2t_loss.data,
            "t2t_nll_loss": t2t_nll_loss.data,
            "src_ntokens": sample["src_ntokens"],
            "tgt_ntokens": sample["tgt_ntokens"],
            "ntokens": sample["src_ntokens"] + sample["tgt_ntokens"],
            "nsentences": sample["nsentences"],
            "src_sample_size": src_sample_size,
            "tgt_sample_size": tgt_sample_size,
        }
        loss = s2s_loss + t2t_loss
        sample_size = src_sample_size + tgt_sample_size
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        src_lprobs, tgt_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        src_lprobs = src_lprobs.view(-1, src_lprobs.size(-1))
        tgt_lprobs = tgt_lprobs.view(-1, tgt_lprobs.size(-1))
        source = model.get_sources(sample, net_output).view(-1, 1)
        target = model.get_targets(sample, net_output).view(-1, 1)
        s2s_loss, s2s_nll_loss = label_smoothed_nll_loss(
            src_lprobs,
            source,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        t2t_loss, t2t_nll_loss = label_smoothed_nll_loss(
            tgt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return s2s_loss, s2s_nll_loss, t2t_loss, t2t_nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        s2s_loss_sum = utils.item(sum(log.get("s2s_loss", 0) for log in logging_outputs))
        s2s_nll_loss_sum = utils.item(sum(log.get("s2s_nll_loss", 0) for log in logging_outputs))
        src_ntokens = utils.item(sum(log.get("src_ntokens", 0) for log in logging_outputs))
        src_sample_size = utils.item(sum(log.get("src_sample_size", 0) for log in logging_outputs))
        t2t_loss_sum = utils.item(sum(log.get("t2t_loss", 0) for log in logging_outputs))
        t2t_nll_loss_sum = utils.item(sum(log.get("t2t_nll_loss", 0) for log in logging_outputs))
        tgt_ntokens = utils.item(sum(log.get("tgt_ntokens", 0) for log in logging_outputs))
        tgt_sample_size = utils.item(sum(log.get("tgt_sample_size", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", s2s_loss_sum / src_sample_size / math.log(2) + t2t_loss_sum / tgt_sample_size / math.log(2), round=3
        )
        metrics.log_scalar(
            "s2s_loss", s2s_loss_sum / src_sample_size / math.log(2), src_sample_size, round=3
        )
        metrics.log_scalar(
            "t2t_loss", t2t_loss_sum / tgt_sample_size / math.log(2), tgt_sample_size, round=3
        )
        metrics.log_scalar(
            "s2s_nll_loss", s2s_nll_loss_sum / src_ntokens / math.log(2), src_ntokens, round=3
        )
        metrics.log_scalar(
            "t2t_nll_loss", t2t_nll_loss_sum / tgt_ntokens / math.log(2), tgt_ntokens, round=3
        )
        metrics.log_derived(
            "s2s_ppl", lambda meters: utils.get_perplexity(meters["s2s_nll_loss"].avg)
        )
        metrics.log_derived(
            "t2t_ppl", lambda meters: utils.get_perplexity(meters["t2t_nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("bidirectional_label_smoothed_cross_entropy_with_mse")
class BidirectionalLabelSmoothedCrossEntropyCriterionWithMSE(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task, sentence_avg, label_smoothing)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample["net_input"]["train_mode"] = 1
        net_output = model(**sample["net_input"])
        s2t_loss, s2t_nll_loss, t2s_loss, t2s_nll_loss, mse_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        src_sample_size = (
            sample["source"].size(0) if self.sentence_avg else sample["src_ntokens"]
        )
        tgt_sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["tgt_ntokens"]
        )
        logging_output = {
            "s2t_loss": s2t_loss.data,
            "s2t_nll_loss": s2t_nll_loss.data,
            "t2s_loss": t2s_loss.data,
            "t2s_nll_loss": t2s_nll_loss.data,
            "mse_loss": mse_loss.data,
            "src_ntokens": sample["src_ntokens"],
            "tgt_ntokens": sample["tgt_ntokens"],
            "ntokens": sample["src_ntokens"] + sample["tgt_ntokens"],
            "nsentences": sample["nsentences"],
            "src_sample_size": src_sample_size,
            "tgt_sample_size": tgt_sample_size,
        }
        loss = s2t_loss + t2s_loss + mse_loss
        sample_size = src_sample_size + tgt_sample_size
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        src_lprobs, tgt_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        src_lprobs = src_lprobs.view(-1, src_lprobs.size(-1))
        tgt_lprobs = tgt_lprobs.view(-1, tgt_lprobs.size(-1))
        source = model.get_sources(sample, net_output).view(-1, 1)
        target = model.get_targets(sample, net_output).view(-1, 1)
        s2t_loss, s2t_nll_loss = label_smoothed_nll_loss(
            tgt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        t2s_loss, t2s_nll_loss = label_smoothed_nll_loss(
            src_lprobs,
            source,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        src_encoder_out = net_output[2]["encoder_out"][0].permute(1, 0, 2)
        src_len = src_encoder_out.size()[1]
        tgt_encoder_out = net_output[3]["encoder_out"][0].permute(1, 0, 2)
        tgt_len = tgt_encoder_out.size()[1]
        if src_len > tgt_len:
            kernel_size = src_len - tgt_len + 1
            src_encoder_out = F.avg_pool1d(src_encoder_out.permute(0, 2, 1), kernel_size=kernel_size, stride=1).permute(0, 2, 1)
        elif src_len < tgt_len:
            kernel_size = tgt_len - src_len + 1
            tgt_encoder_out = F.avg_pool1d(tgt_encoder_out.permute(0, 2, 1), kernel_size=kernel_size, stride=1).permute(0, 2, 1)
        mse_loss = F.mse_loss(src_encoder_out, tgt_encoder_out)
        return s2t_loss, s2t_nll_loss, t2s_loss, t2s_nll_loss, mse_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        s2t_loss_sum = utils.item(sum(log.get("s2t_loss", 0) for log in logging_outputs))
        s2t_nll_loss_sum = utils.item(sum(log.get("s2t_nll_loss", 0) for log in logging_outputs))
        src_ntokens = utils.item(sum(log.get("src_ntokens", 0) for log in logging_outputs))
        src_sample_size = utils.item(sum(log.get("src_sample_size", 0) for log in logging_outputs))
        t2s_loss_sum = utils.item(sum(log.get("t2s_loss", 0) for log in logging_outputs))
        t2s_nll_loss_sum = utils.item(sum(log.get("t2s_nll_loss", 0) for log in logging_outputs))
        tgt_ntokens = utils.item(sum(log.get("tgt_ntokens", 0) for log in logging_outputs))
        tgt_sample_size = utils.item(sum(log.get("tgt_sample_size", 0) for log in logging_outputs))
        mse_loss_sum = utils.item(sum(log.get("mse_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", s2t_loss_sum / tgt_sample_size / math.log(2) + t2s_loss_sum / src_sample_size / math.log(2) + mse_loss_sum, round=3
        )
        metrics.log_scalar(
            "s2t_loss", s2t_loss_sum / tgt_sample_size / math.log(2), tgt_sample_size, round=3
        )
        metrics.log_scalar(
            "t2s_loss", t2s_loss_sum / src_sample_size / math.log(2), src_sample_size, round=3
        )
        metrics.log_scalar(
            "mse_loss", mse_loss_sum, round=3
        )
        metrics.log_scalar(
            "s2t_nll_loss", s2t_nll_loss_sum / tgt_ntokens / math.log(2), tgt_ntokens, round=3
        )
        metrics.log_scalar(
            "t2s_nll_loss", t2s_nll_loss_sum / src_ntokens / math.log(2), src_ntokens, round=3
        )
        metrics.log_derived(
            "s2t_ppl", lambda meters: utils.get_perplexity(meters["s2t_nll_loss"].avg)
        )
        metrics.log_derived(
            "t2s_ppl", lambda meters: utils.get_perplexity(meters["t2s_nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
