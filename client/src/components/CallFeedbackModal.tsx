import React, { useState } from 'react';
import { Star, CheckCircle2, Sparkles, ArrowRight, X } from 'lucide-react';

interface CallFeedbackModalProps {
  isOpen: boolean;
  roomId: string;
  onSubmit: (feedback: { rating: number }) => void;
  onSkip: () => void;
}

const RATING_LABELS: Record<number, { title: string; emoji: string }> = {
  1: { title: 'Poor Experience', emoji: '😕' },
  2: { title: 'Fair', emoji: '😐' },
  3: { title: 'Good', emoji: '🙂' },
  4: { title: 'Very Good', emoji: '😊' },
  5: { title: 'Excellent!', emoji: '🚀' },
};

export const CallFeedbackModal: React.FC<CallFeedbackModalProps> = ({
  isOpen,
  roomId,
  onSubmit,
  onSkip,
}) => {
  const [rating, setRating] = useState<number>(0);
  const [hoverRating, setHoverRating] = useState<number>(0);
  const [isSubmitted, setIsSubmitted] = useState<boolean>(false);

  if (!isOpen) return null;

  const activeRating = hoverRating || rating;
  const ratingInfo = activeRating > 0 ? RATING_LABELS[activeRating] : null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (rating === 0) return;
    setIsSubmitted(true);
    setTimeout(() => {
      onSubmit({ rating });
    }, 700);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/30 backdrop-blur-sm animate-fade-in font-sans">
      <div className="relative w-full max-w-md bg-white border border-slate-200/80 rounded-3xl p-6 sm:p-8 shadow-xl shadow-slate-200/50 text-slate-900 overflow-hidden">
        {/* Subtle decorative glow */}
        <div className="absolute -top-20 -right-20 w-40 h-40 bg-blue-50 rounded-full blur-2xl pointer-events-none" />

        {isSubmitted ? (
          <div className="py-8 flex flex-col items-center justify-center text-center animate-scale-up">
            <div className="h-16 w-16 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4 border border-emerald-200/70 shadow-sm">
              <CheckCircle2 className="h-8 w-8 stroke-[2.5]" />
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-1.5">
              Thank you for your rating!
            </h3>
            <p className="text-xs sm:text-sm text-slate-500">
              Your feedback helps us continuously enhance call quality.
            </p>
          </div>
        ) : (
          <div>
            {/* Header */}
            <div className="flex items-start justify-between mb-6">
              <div>
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 border border-blue-200/60 text-blue-600 text-xs font-semibold mb-2.5">
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Call Ended • Room {roomId}</span>
                </div>
                <h3 className="text-2xl font-bold text-slate-900 tracking-tight">
                  How was your call?
                </h3>
                <p className="text-xs sm:text-sm text-slate-500 mt-1">
                  Rate your overall call quality and experience.
                </p>
              </div>

              <button
                onClick={onSkip}
                className="p-1.5 rounded-full text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
                title="Close"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* 5-Star Interactive Rating Only */}
            <div className="flex flex-col items-center justify-center py-6 bg-slate-50/80 rounded-2xl border border-slate-100 mb-6">
              <div className="flex items-center gap-2.5 sm:gap-3">
                {[1, 2, 3, 4, 5].map((star) => {
                  const isFilled = (hoverRating || rating) >= star;
                  return (
                    <button
                      key={star}
                      type="button"
                      onClick={() => setRating(star)}
                      onMouseEnter={() => setHoverRating(star)}
                      onMouseLeave={() => setHoverRating(0)}
                      className="group p-1.5 focus:outline-none transition-transform hover:scale-115 active:scale-95 cursor-pointer"
                      title={`${star} Star${star > 1 ? 's' : ''}`}
                    >
                      <Star
                        className={`h-9 w-9 sm:h-10 sm:w-10 transition-all duration-200 ${
                          isFilled
                            ? 'text-amber-400 fill-amber-400 drop-shadow-[0_2px_8px_rgba(251,191,36,0.4)]'
                            : 'text-slate-300 hover:text-amber-300'
                        }`}
                      />
                    </button>
                  );
                })}
              </div>

              {/* Dynamic Rating description */}
              <div className="h-7 mt-3 flex items-center justify-center text-center">
                {ratingInfo ? (
                  <div className="animate-fade-in flex items-center gap-1.5">
                    <span className="text-lg">{ratingInfo.emoji}</span>
                    <span className="text-sm font-semibold text-slate-800">
                      {ratingInfo.title}
                    </span>
                  </div>
                ) : (
                  <span className="text-xs text-slate-400 font-medium">
                    Tap a star to rate (1 to 5 stars)
                  </span>
                )}
              </div>
            </div>

            {/* Actions: Blue Primary Button matching Image 2 */}
            <div className="flex flex-col items-center gap-3">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={rating === 0}
                className={`w-full py-3.5 px-5 rounded-2xl font-semibold text-sm flex items-center justify-center gap-2 transition-all cursor-pointer ${
                  rating > 0
                    ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-500/20 active:scale-[0.99]'
                    : 'bg-slate-100 text-slate-400 cursor-not-allowed shadow-none'
                }`}
              >
                <span>Submit Rating</span>
                <ArrowRight className="h-4 w-4" />
              </button>

              <button
                type="button"
                onClick={onSkip}
                className="text-xs font-medium text-slate-400 hover:text-slate-600 py-1 transition-colors cursor-pointer"
              >
                Skip for now
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
