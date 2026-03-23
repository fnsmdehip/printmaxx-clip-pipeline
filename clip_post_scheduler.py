#!/usr/bin/env python3
"""
Clip Post Scheduler - Generate optimal posting schedules for viral clips.

Takes clips metadata CSV and creates posting schedule with optimal times
for TikTok (9am, 12pm, 5pm, 7pm) and Twitter (8am, 12pm, 5pm, 9pm).

Usage:
    python3 clip_post_scheduler.py --input clips/clips_metadata.csv --output posting_schedule.csv
    python3 clip_post_scheduler.py --input clips/clips_metadata.csv --days 14 --accounts accounts.json
"""

import csv
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json


class PostScheduler:
    """Generate optimal posting schedules for viral clips."""

    # Platform optimal times (hour of day)
    OPTIMAL_TIMES = {
        'tiktok': [9, 12, 17, 19],    # 9am, 12pm, 5pm, 7pm
        'twitter': [8, 12, 17, 21],   # 8am, 12pm, 5pm, 9pm
        'instagram': [9, 12, 17, 19], # 9am, 12pm, 5pm, 7pm
        'youtube': [14, 17, 20],      # 2pm, 5pm, 8pm
    }

    PLATFORM_ALIASES = {
        'x': 'twitter',
        'ig': 'instagram',
        'tik': 'tiktok',
        'yt': 'youtube',
    }

    def __init__(self, accounts: Dict[str, List[str]] = None):
        """
        Initialize scheduler.

        Args:
            accounts: Dict mapping platform to list of account handles
                     e.g., {'tiktok': ['@faithaccount'], 'twitter': ['@printmaxxer']}
        """
        self.accounts = accounts or self._default_accounts()

    def _default_accounts(self) -> Dict[str, List[str]]:
        """Default accounts if none provided."""
        return {
            'tiktok': ['@printmaxxer'],
            'twitter': ['@printmaxxer'],
            'instagram': ['@printmaxxer'],
            'youtube': ['@printmaxxer'],
        }

    def load_clips_metadata(self, csv_path: Path) -> List[Dict]:
        """Load clips from metadata CSV."""
        clips = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    clips.append(row)
            print(f"✅ Loaded {len(clips)} clips from {csv_path}")
            return clips
        except FileNotFoundError:
            print(f"❌ File not found: {csv_path}")
            return []
        except Exception as e:
            print(f"❌ Error loading clips: {e}")
            return []

    def generate_schedule(
        self,
        clips: List[Dict],
        days: int = 7,
        start_date: datetime = None,
        platforms: List[str] = None
    ) -> List[Dict]:
        """
        Generate posting schedule for clips.

        Args:
            clips: List of clip metadata dicts
            days: Number of days to schedule across
            start_date: Start date (defaults to tomorrow)
            platforms: List of platforms to post to (defaults to all)

        Returns:
            List of scheduled post dicts
        """
        if not clips:
            return []

        start_date = start_date or (datetime.now() + timedelta(days=1))
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        platforms = platforms or ['tiktok', 'twitter']
        platforms = [self._normalize_platform(p) for p in platforms]

        # Sort clips by viral score (highest first)
        clips_sorted = sorted(
            clips,
            key=lambda x: float(x.get('viral_score', 0)),
            reverse=True
        )

        schedule = []

        # Generate time slots
        time_slots = self._generate_time_slots(start_date, days, platforms)

        # Assign clips to time slots
        # Round-robin across platforms to distribute content evenly
        clip_idx = 0
        for slot in time_slots:
            if clip_idx >= len(clips_sorted):
                break

            clip = clips_sorted[clip_idx]

            # Select account (round-robin if multiple)
            accounts = self.accounts.get(slot['platform'], ['@default'])
            account = accounts[clip_idx % len(accounts)]

            # Generate post text
            post_text = self._generate_post_text(clip, slot['platform'])

            schedule.append({
                'clip_id': clip['clip_id'],
                'post_text': post_text,
                'media_path': clip['output_path'],
                'platform': slot['platform'],
                'account': account,
                'scheduled_time': slot['time'].isoformat(),
                'viral_score': clip.get('viral_score', ''),
                'caption': clip.get('caption_text', ''),
            })

            clip_idx += 1

        print(f"✅ Generated {len(schedule)} scheduled posts across {days} days")
        return schedule

    def _generate_time_slots(
        self,
        start_date: datetime,
        days: int,
        platforms: List[str]
    ) -> List[Dict]:
        """Generate list of posting time slots."""
        slots = []

        for day_offset in range(days):
            date = start_date + timedelta(days=day_offset)

            for platform in platforms:
                optimal_hours = self.OPTIMAL_TIMES.get(platform, [9, 12, 17])

                for hour in optimal_hours:
                    slots.append({
                        'platform': platform,
                        'time': date.replace(hour=hour, minute=0)
                    })

        return slots

    def _normalize_platform(self, platform: str) -> str:
        """Normalize platform name."""
        platform = platform.lower()
        return self.PLATFORM_ALIASES.get(platform, platform)

    def _generate_post_text(self, clip: Dict, platform: str) -> str:
        """Generate post text for clip based on platform."""
        caption = clip.get('caption_text', 'Check this out')

        # Platform-specific formatting
        if platform == 'twitter':
            # Twitter: Keep it punchy
            return caption[:280]  # Twitter character limit

        elif platform == 'tiktok':
            # TikTok: Add hashtags
            base = caption
            hashtags = self._generate_hashtags(clip, count=5)
            return f"{base}\n\n{' '.join(hashtags)}"

        elif platform == 'instagram':
            # Instagram: More hashtags
            base = caption
            hashtags = self._generate_hashtags(clip, count=10)
            return f"{base}\n\n{' '.join(hashtags)}"

        elif platform == 'youtube':
            # YouTube Shorts: Title style
            return caption

        return caption

    def _generate_hashtags(self, clip: Dict, count: int = 5) -> List[str]:
        """Generate hashtags based on clip content."""
        # Basic hashtags (could be enhanced with AI)
        base_tags = ['#viral', '#fyp', '#trending', '#shorts']

        # Add some variation
        viral_score = float(clip.get('viral_score', 5))
        if viral_score >= 8:
            base_tags.extend(['#mustwatch', '#insane'])
        elif viral_score >= 6:
            base_tags.extend(['#interesting', '#wow'])

        return base_tags[:count]

    def save_schedule(self, schedule: List[Dict], output_path: Path):
        """Save schedule to CSV."""
        if not schedule:
            print("⚠️  No schedule to save")
            return

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'clip_id', 'post_text', 'media_path', 'platform',
                'account', 'scheduled_time', 'viral_score', 'caption'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(schedule)

        print(f"✅ Saved schedule to {output_path}")

    def print_summary(self, schedule: List[Dict]):
        """Print schedule summary."""
        if not schedule:
            return

        print(f"\n{'='*60}")
        print("POSTING SCHEDULE SUMMARY")
        print(f"{'='*60}\n")

        # Group by platform
        by_platform = {}
        for post in schedule:
            platform = post['platform']
            if platform not in by_platform:
                by_platform[platform] = []
            by_platform[platform].append(post)

        for platform, posts in by_platform.items():
            print(f"{platform.upper()}: {len(posts)} posts")

            # Show first 3
            for post in posts[:3]:
                time = datetime.fromisoformat(post['scheduled_time'])
                print(f"  - {time.strftime('%b %d, %I:%M%p')}: {post['post_text'][:50]}...")

            if len(posts) > 3:
                print(f"  ... and {len(posts) - 3} more")
            print()

        print(f"Total posts: {len(schedule)}")
        print(f"First post: {datetime.fromisoformat(schedule[0]['scheduled_time']).strftime('%b %d, %Y at %I:%M%p')}")
        print(f"Last post: {datetime.fromisoformat(schedule[-1]['scheduled_time']).strftime('%b %d, %Y at %I:%M%p')}")
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimal posting schedule for viral clips"
    )

    parser.add_argument('--input', required=True, help='Input clips metadata CSV')
    parser.add_argument('--output', default='posting_schedule.csv', help='Output schedule CSV')
    parser.add_argument('--days', type=int, default=7, help='Days to schedule across (default: 7)')
    parser.add_argument('--platforms', nargs='+', default=['tiktok', 'twitter'],
                       help='Platforms to post to (default: tiktok twitter)')
    parser.add_argument('--accounts', help='JSON file with account mapping')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD), defaults to tomorrow')

    args = parser.parse_args()

    # Load accounts if provided
    accounts = None
    if args.accounts:
        try:
            with open(args.accounts, 'r') as f:
                accounts = json.load(f)
            print(f"✅ Loaded accounts from {args.accounts}")
        except Exception as e:
            print(f"⚠️  Could not load accounts: {e}, using defaults")

    # Parse start date
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"⚠️  Invalid date format, using default (tomorrow)")

    # Initialize scheduler
    scheduler = PostScheduler(accounts)

    # Load clips
    clips = scheduler.load_clips_metadata(Path(args.input))
    if not clips:
        print("❌ No clips to schedule")
        return

    # Generate schedule
    schedule = scheduler.generate_schedule(
        clips,
        days=args.days,
        start_date=start_date,
        platforms=args.platforms
    )

    if not schedule:
        print("❌ Could not generate schedule")
        return

    # Save schedule
    scheduler.save_schedule(schedule, Path(args.output))

    # Print summary
    scheduler.print_summary(schedule)

    print(f"✅ Schedule ready for import to Buffer/Publer/Hootsuite")
    print(f"   Import file: {args.output}")


if __name__ == '__main__':
    main()
